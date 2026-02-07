import json
import sys
import linecache
import random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import tqdm
import hashlib
import os
import shutil
import concurrent.futures
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset, ConcatDataset
from transformers import AutoTokenizer


class SequentialDataset(Dataset):
    ''' 通用顺序数据集 (支持 Parquet & JSONL 混合读取)
    功能：
    1. 递归扫描目录下所有 .parquet 和 .jsonl 文件。
    2. 根据 data_config 中的关键字匹配路径，自动应用列映射。
    3. 支持 FIM、CoT 自动提取 (支持 startswith 匹配)、工具调用解析。
    4. 支持全量内存加载 (In-Memory)。
    Args:
        root_dir (str | Path): 数据集根目录。
        tokenizer: 分词器。
        config (dict): 文件夹名到列名的映射配置。
        cot_delimiter (tuple, optional): CoT 提取符号 ('<think>', '</think>')。
        fim_rate (float): FIM 概率。
        max_length (int, optional): 最大长度。
        in_memory (bool): 是否预加载到内存。
        remove_strings (list[str], optional): 在 model_text 最终阶段删除的字符串列表。
    '''
    def __init__(
        self, 
        root_dir, 
        tokenizer, 
        config=None, 
        cot_delimiter=None, 
        fim_rate=0.5, 
        max_length=None, 
        in_memory=False,
        remove_strings=None
    ):
        self.root_dir = Path(root_dir)
        
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
            
        self.config = config or {}
        
        self.cot_start = None
        self.cot_end = None
        if cot_delimiter and len(cot_delimiter) == 2:
            self.cot_start, self.cot_end = cot_delimiter
            
        self.fim_rate = fim_rate
        self.max_length = max_length
        self.in_memory = in_memory
        self.remove_strings = remove_strings
        
        self.data_entries = []
        self.total_rows = 0
        self._build_index()
    def _get_mapping_for_path(self, file_path):
        ''' 根据路径关键字匹配配置 '''
        path_str = str(file_path).replace('\\', '/')
        for folder_key, mapping in self.config.items():
            if folder_key in path_str:
                return mapping
        
        return {
            'system': 'system',
            'train': 'text',
            'user': 'user',
            'model': 'model',
            'reasoning': 'thought',
            'tool_calls': 'tool_calls',
            'tool_result': 'tool_output'
        }
    def _get_file_type(self, path):
        ext = path.suffix.lower()
        if ext == '.parquet': return 'parquet'
        if ext == '.jsonl': return 'jsonl'
        return None
    
    def _build_index(self):
        print(f'[SequentialDataset] Scanning {self.root_dir} (In-Memory: {self.in_memory})...')
        
        files = []
        
        if self.config:
            print(f"   - Targeted Scan: {list(self.config.keys())}")
            
            for folder_name in self.config.keys():
                target_path = self.root_dir / folder_name
                
                if not target_path.exists():
                    print(f"   ⚠️ Warning: Config path '{folder_name}' not found under root.")
                    continue
                
                if target_path.is_dir():
                    for ext in ['*.parquet', '*.jsonl']:
                        files.extend(list(target_path.rglob(ext)))
                elif target_path.is_file():
                     files.append(target_path)
                     
        else:
            print(f"   - Full Scan (No config provided)")
            for ext in ['*.parquet', '*.jsonl']:
                files.extend(list(self.root_dir.rglob(ext)))
        
        files = sorted(list(set(files)))

        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc="Indexing Files")
        except ImportError:
            iterator = files

        for file_path in iterator:
            ftype = self._get_file_type(file_path)
            mapping = self._get_mapping_for_path(file_path)
            if not mapping: continue
            
            cols = [v for v in mapping.values() if v]
            num_rows = 0
            cached_data = None

            try:
                if ftype == 'parquet':
                    if self.in_memory and cols:
                        cached_data = pd.read_parquet(file_path, columns=cols, engine='pyarrow')
                        num_rows = len(cached_data)
                    else:
                        meta = pq.read_metadata(file_path)
                        num_rows = meta.num_rows

                elif ftype == 'jsonl':
                    if self.in_memory:
                        cached_data = pd.read_json(file_path, lines=True)
                        num_rows = len(cached_data)
                    else:
                        count = 0
                        with open(file_path, 'rb') as f:
                            while True:
                                buffer = f.read(1024*1024)
                                if not buffer: break
                                count += buffer.count(b'\n')
                        num_rows = count

            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                continue

            if num_rows == 0: continue

            self.data_entries.append({
                'path': str(file_path),
                'type': ftype,
                'start_idx': self.total_rows,
                'end_idx': self.total_rows + num_rows,
                'mapping': mapping,
                'cache': cached_data
            })
            self.total_rows += num_rows
            
        print(f'[SequentialDataset] Indexed {len(self.data_entries)} files, {self.total_rows} rows.')

    def __len__(self):
        return self.total_rows
    def _read_row(self, entry, local_idx):
        if self.in_memory and entry['cache'] is not None:
            try: return entry['cache'].iloc[local_idx]
            except: return {}
        ftype = entry['type']
        cols = [v for v in entry['mapping'].values() if v]
        if ftype == 'parquet':
            try:
                return pd.read_parquet(entry['path'], columns=cols, engine='pyarrow').iloc[local_idx]
            except: return {}
        elif ftype == 'jsonl':
            try:
                line = linecache.getline(entry['path'], local_idx + 1)
                if line: return json.loads(line)
            except: return {}
        
        return {}
    def _create_fim_dict(self, text):
        if len(text) < 50: return None
        total_len = len(text)
        span_len = int(total_len * np.random.uniform(0.1, 0.3))
        if span_len < 5: return None
        start = np.random.randint(0, total_len - span_len)
        return {
            'role': 'fim', 'prefix': text[:start], 'middle': text[start:start+span_len], 'suffix': text[start+span_len:]
        }
    def _safe_json_load(self, content):
        if not content: return None
        if isinstance(content, (list, dict)): return content
        try: return json.loads(str(content))
        except: return None
        
    def _extract_embedded_cot(self, text):
        ''' 根据 startswith 判断并提取 CoT '''
        if not self.cot_start or not text:
            return text, None
        
        processed_text = text.lstrip()
        
        if processed_text.startswith(self.cot_start):
            start_offset = len(self.cot_start)
            e_idx = processed_text.find(self.cot_end, start_offset)
            
            if e_idx != -1:
                cot_content = processed_text[start_offset : e_idx].strip()
                response_content = processed_text[e_idx + len(self.cot_end) :].strip()
                return response_content, cot_content
        
        return text, None
    
    def get_messages(self, idx):
        entry = next(e for e in self.data_entries if e['start_idx'] <= idx < e['end_idx'])
        mapping = entry['mapping']
        
        row_data = self._read_row(entry, idx - entry['start_idx'])
        
        conversations = None
        if isinstance(row_data, dict):
            conversations = row_data.get('conversations')
        elif hasattr(row_data, 'get'):
            conversations = row_data.get('conversations')
        elif hasattr(row_data, 'index') and 'conversations' in row_data.index:
            conversations = row_data['conversations']
            
        messages = []
        if conversations and isinstance(conversations, list):
            for msg in conversations:
                role = msg.get('role')
                content = msg.get('content')
                if not role or content is None: continue
                
                if role == 'assistant':
                    model_text = content
                    reasoning_text = None
                    
                    if self.cot_start:
                        model_text, parsed_cot = self._extract_embedded_cot(model_text)
                        if parsed_cot:
                            reasoning_text = parsed_cot
                    
                    if self.remove_strings:
                        for s in self.remove_strings:
                            model_text = model_text.replace(s, '')
                        model_text = model_text.strip()
                        
                    messages.append({
                        'role': 'model',
                        'content': model_text,
                        'reasoning_content': reasoning_text if reasoning_text else ''
                    })
                else:
                    messages.append({'role': role, 'content': content})
        else:
            def get(key):
                col_name = mapping.get(key)
                if not col_name: return None
                
                val = None
                if hasattr(row_data, 'get'):
                    val = row_data.get(col_name)
                elif hasattr(row_data, 'index') and col_name in row_data.index:
                    val = row_data[col_name]
                
                if pd.notna(val) and val is not None:
                    s = str(val).strip()
                    return s if len(s) > 0 else None
                return None
            
            sys_text = get('system')
            if sys_text: messages.append({'role': 'system', 'content': sys_text})
            train_text = get('train')
            if train_text:
                if self.fim_rate > 0 and np.random.rand() < self.fim_rate:
                    fim_msg = self._create_fim_dict(train_text)
                    messages.append(fim_msg if fim_msg else {'role': 'train', 'content': train_text})
                else:
                    messages.append({'role': 'train', 'content': train_text})
            user_text = get('user')
            if user_text: messages.append({'role': 'user', 'content': user_text})
            model_text = get('model')
            reasoning_text = get('reasoning')
            
            if model_text and self.cot_start:
                parsed_model, parsed_cot = self._extract_embedded_cot(model_text)
                if parsed_cot:
                    model_text = parsed_model
                    if not reasoning_text: reasoning_text = parsed_cot
            
            if model_text and self.remove_strings:
                for s in self.remove_strings:
                    model_text = model_text.replace(s, '')
                model_text = model_text.strip()
                
            tool_calls_raw = get('tool_calls')
            if model_text or reasoning_text or tool_calls_raw:
                msg = {
                    'role': 'model',
                    'content': model_text if model_text else '',
                    'reasoning_content': reasoning_text if reasoning_text else ''
                }
                if tool_calls_raw:
                    parsed_tools = self._safe_json_load(tool_calls_raw)
                    if parsed_tools: msg['tool_calls'] = parsed_tools
                messages.append(msg)
            tool_res = get('tool_result')
            if tool_res: messages.append({'role': 'tool', 'content': tool_res})
        
        return messages

    def __getitem__(self, idx):
        messages = self.get_messages(idx)
        if not messages:
            return {
                'input_ids': torch.tensor([], dtype=torch.long),
                'labels': torch.tensor([], dtype=torch.long)
            }
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            truncation=False,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return tokenized[0]


class SFTDataset(SequentialDataset):
    ''' SFT 数据集 (Masked Supervised Fine-Tuning)
    继承自 SequentialDataset，但在 __getitem__ 时：
    1. 计算完整的 input_ids。
    2. 通过增量 apply_chat_template 识别 model 回复片段。
    3. 生成 mask (1 for model tokens, 0 for others)。
    '''
    def __getitem__(self, idx):
        messages = self.get_messages(idx)
        if not messages:
            return {
                'input_ids': torch.tensor([], dtype=torch.long),
                'mask': torch.tensor([], dtype=torch.long)
            }   

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )[0].ids
        
        mask = torch.zeros(len(input_ids), dtype=torch.long)
        
        curr_len = 0
        for i, msg in enumerate(messages):
            part_ids = self.tokenizer.apply_chat_template(
                messages[:i+1],
                truncation=False,
                padding=False,
                return_tensors='pt'
            )[0]
            
            end_len = len(part_ids)
            
            if end_len > len(input_ids):
                end_len = len(input_ids)
                
            if msg['role'] == 'model':
                mask[curr_len:end_len] = 1
            
            curr_len = end_len
            if curr_len >= len(input_ids):
                break
                
        if self.max_length and len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            mask = mask[:self.max_length]
            
        return {
            'input_ids': input_ids,
            'mask': mask
        }


class CachedDataset(Dataset):
    ''' 缓存加速数据集 (支持增量更新 & 断点续传)
    '''
    def __init__(
        self, 
        source_dataset, 
        cache_dir, 
        max_length=2048, 
        chunk_size=1024*1024*1024, # 1GB
        rebuild=False, 
        num_workers=8,
        save_interval=10000, # 每处理多少条保存一次 Index
        force_reuse=False
    ):
        self.source_dataset = source_dataset
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        self.save_interval = save_interval
        self.meta: dict = {}
        
        self.meta_path = os.path.join(cache_dir, "meta.json")
        self.idx_path = os.path.join(cache_dir, "index.npy")
        self.data_prefix = "data_"
        if rebuild:
            print("🔄 [CachedDataset] Rebuild requested. Clearing old cache...")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            self._build_cache(resume_from=0)
        else:
            processed_count = self._get_processed_count()
            total_count = len(self.source_dataset)
            
            if processed_count < total_count:
                print(f"🔄 [CachedDataset] Incremental Update: {processed_count} -> {total_count}")
                self._build_cache(resume_from=processed_count)
            elif processed_count > total_count:
                if force_reuse:
                    print(f"⚠️ [CachedDataset] Cache ({processed_count}) > Source ({total_count}). Source shrunk but force_reuse=True. Using existing cache.")
                else:
                    print(f"⚠️ [CachedDataset] Cache ({processed_count}) > Source ({total_count}). Source may have shrunk? Rebuilding...")
                    shutil.rmtree(cache_dir)
                    self._build_cache(resume_from=0)
            else:
                print(f"✅ [CachedDataset] Cache up-to-date ({processed_count} samples).")
    
        self._load_cache()

    def _get_processed_count(self):
        ''' 获取当前已缓存的样本数量 '''
        if not os.path.exists(self.idx_path) or not os.path.exists(self.meta_path):
            return 0
        try:
            indices = np.load(self.idx_path, mmap_mode='r')
            return len(indices)
        except:
            return 0
        
    def _process_sample(self, idx):
        ''' 子线程：处理单个样本 '''
        try:
            raw_sample = self.source_dataset[idx]
            
            ids = raw_sample
            if hasattr(raw_sample, 'ids'): ids = raw_sample.ids
            elif isinstance(raw_sample, dict) and 'input_ids' in raw_sample: ids = raw_sample['input_ids']
            
            if isinstance(ids, torch.Tensor): ids = ids.tolist()
            
            if not ids or len(ids) == 0: return None
            if len(ids) > self.max_length: ids = ids[:self.max_length]
            return np.array(ids, dtype=self.dtype).tobytes()
        except Exception as e:
            return f"Error: {str(e)[:100]}"
        
    def _build_cache(self, resume_from=0):
        ''' 构建/更新缓存的核心逻辑 '''
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        vocab_size = getattr(getattr(self.source_dataset, 'tokenizer', None), "vocab_size", 65536)
        self.dtype = np.uint16 if vocab_size < 65535 else np.uint32
        dtype_name = np.dtype(self.dtype).name
        
        total_samples = len(self.source_dataset)
        
        indices = []
        current_file_id = 0
        current_file_ptr = 0
        total_tokens = 0
        
        if resume_from > 0:
            indices = np.load(self.idx_path).tolist()
            
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
                total_tokens = meta.get('total_tokens', 0)
            
            if indices:
                last_entry = indices[-1]
                current_file_id = int(last_entry[0])
                current_file_ptr = int(last_entry[1]) + int(last_entry[2]) * np.dtype(self.dtype).itemsize
                
                last_bin_path = os.path.join(self.cache_dir, f"{self.data_prefix}{current_file_id}.bin")
                if os.path.exists(last_bin_path):
                    real_size = os.path.getsize(last_bin_path)
                    if real_size > current_file_ptr:
                        print(f"🔧 Repairing bin file: Truncating {last_bin_path} from {real_size} to {current_file_ptr}")
                        with open(last_bin_path, "r+b") as f:
                            f.truncate(current_file_ptr)
                            
        print(f"📦 [CachedDataset] {'Resuming' if resume_from > 0 else 'Starting'} Cache Build...")
        print(f"   - Progress: {resume_from} -> {total_samples}")
        print(f"   - File ID:  {current_file_id} (Offset: {current_file_ptr})")
        sys.stdout.flush()
        bin_path = os.path.join(self.cache_dir, f"{self.data_prefix}{current_file_id}.bin")
        mode = "ab" if os.path.exists(bin_path) else "wb"
        f_current = open(bin_path, mode)
        
        try:
            if self.num_workers <= 1:
                iterable = map(self._process_sample, range(resume_from, total_samples))
            else:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)
                iterable = executor.map(self._process_sample, range(resume_from, total_samples), chunksize=1)

            pbar = tqdm.tqdm(iterable, total=total_samples - resume_from, desc="Caching", unit="it", ncols=100)
            
            step_count = 0
            
            for res in pbar:
                step_count += 1
                
                if res is None: continue
                if isinstance(res, str) and res.startswith("Error"): continue
                
                byte_len = len(res)
                item_len = byte_len // np.dtype(self.dtype).itemsize
                
                if (current_file_ptr + byte_len > self.chunk_size) and (current_file_ptr > 0):
                    f_current.close()
                    current_file_id += 1
                    current_file_ptr = 0
                    path = os.path.join(self.cache_dir, f"{self.data_prefix}{current_file_id}.bin")
                    f_current = open(path, "wb")
                
                f_current.write(res)
                indices.append((current_file_id, current_file_ptr, item_len))
                
                current_file_ptr += byte_len
                total_tokens += item_len
                
                if step_count % 1000 == 0:
                    pbar.set_postfix({"f": current_file_id, "tok": f"{total_tokens/1e6:.1f}M"})
                
                if step_count % self.save_interval == 0:
                    self._save_metadata(indices, total_tokens, dtype_name, current_file_id)
            
            if self.num_workers > 1:
                executor.shutdown()
        finally:
            if f_current: f_current.close()
            
        print("\n📝 Saving final index...")
        self._save_metadata(indices, total_tokens, dtype_name, current_file_id)
        print(f"✅ Build Complete! Total: {len(indices)} samples.")

    def _save_metadata(self, indices, total_tokens, dtype_name, current_file_id):
        ''' 保存索引和元数据 (Helper) '''
        np_indices = np.array(indices, dtype=np.uint64)
        np.save(self.idx_path, np_indices)
        
        meta = {
            "dtype": dtype_name,
            "total_samples": len(indices),
            "total_tokens": int(total_tokens),
            "max_length": self.max_length,
            "num_chunks": current_file_id + 1
        }

        temp_path = self.meta_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(meta, f)
        os.replace(temp_path, self.meta_path)

    def _load_cache(self):
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        
        self.dtype = np.dtype(self.meta["dtype"])
        self.num_chunks = self.meta.get("num_chunks", 1)
        
        idx_candidates = ["index.npy", "index.bin", "index.bin.npy"]
        real_idx_path = next((os.path.join(self.cache_dir, p) for p in idx_candidates if os.path.exists(os.path.join(self.cache_dir, p))), None)
        
        if not real_idx_path: raise FileNotFoundError(f"Missing index in {self.cache_dir}")
        self.indices = np.load(real_idx_path, mmap_mode='r')
        self.num_samples = len(self.indices)
        
        self.mmaps = []
        for i in range(self.num_chunks):
            path = os.path.join(self.cache_dir, f"{self.data_prefix}{i}.bin")
            if os.path.exists(path):
                self.mmaps.append(np.memmap(path, dtype=self.dtype, mode='r'))
            else:
                self.mmaps.append(None)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        file_id, byte_offset, length = self.indices[idx]
        item_offset = int(byte_offset // np.dtype(self.dtype).itemsize)
        mmap = self.mmaps[int(file_id)]
        if mmap is None: return torch.tensor([], dtype=torch.long)
        
        np_array = np.array(mmap[item_offset : item_offset + int(length)], copy=True)
        return torch.from_numpy(np_array.astype(np.int64))
    
    def get_sample_length(self, idx):
        ''' 快速获取样本长度 (仅读取索引，不加载数据) '''
        try:
            return int(self.indices[idx][2])
        except:
            return 0


class AutoMixedDataset(ConcatDataset):
    ''' 自动混合数据集 (解决增量缓存问题)
    
    原理：
    不将所有数据混在一起缓存，而是根据 config 的 key 为每个子数据集
    创建独立的 CachedDataset。
    
    优势：
    1. 修改 config 删除某个数据集时，不需要重建其他数据集的缓存。
    2. 支持单独复用某个子数据集的缓存。
    '''
    def __init__(
        self, 
        root_dir, 
        tokenizer, 
        data_config, 
        cache_root_dir,
        cot_delimiter=None,
        fim_rate=0.5,
        in_memory=True,
        max_length=4096,
        num_workers=8,
        remove_strings=None,
        force_reuse=False
    ):
        self.datasets = []
        self.configs = []
        
        for name, sub_mapping in data_config.items():
            
            print(f"🔗 [AutoMixed] Preparing subset: {name}")
            
            sub_config = {name: sub_mapping}
            
            # 包含 remove_strings 到 hash 计算中，确保配置变更时重建缓存
            hash_data = {
                'config': sub_config,
                'remove_strings': remove_strings
            }
            config_str = json.dumps(hash_data, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            
            sub_cache_dir = os.path.join(cache_root_dir, f"{name}_{config_hash}")
            
            if force_reuse and not os.path.exists(sub_cache_dir) and os.path.exists(cache_root_dir):
                candidates = []
                for entry in os.scandir(cache_root_dir):
                    if entry.is_dir() and entry.name.startswith(f"{name}_"):
                        candidates.append(entry.path)
                
                if candidates:
                    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    old_cache = candidates[0]
                    print(f"⚠️ [AutoMixed] Force reuse active: Redirecting '{name}' to existing cache '{os.path.basename(old_cache)}' (Expected hash: {config_hash})")
                    sub_cache_dir = old_cache
            
            raw_dataset = SequentialDataset(
                root_dir=root_dir,
                tokenizer=tokenizer,
                config=sub_config,
                cot_delimiter=cot_delimiter,
                fim_rate=fim_rate,
                max_length=max_length,
                in_memory=in_memory,
                remove_strings=remove_strings
            )
            
            cached_ds = CachedDataset(
                source_dataset=raw_dataset,
                cache_dir=sub_cache_dir,
                max_length=max_length,
                num_workers=num_workers,
                save_interval=10000,
                force_reuse=force_reuse
            )
            
            self.datasets.append(cached_ds)
        
        super().__init__(self.datasets)


class PackedSeqDataset(IterableDataset):
    ''' 打包序列数据集 (Sequence Packing)
    将 SequentialParquetDataset 中变长的样本拼接起来，
    并按照固定的 max_length 进行切割，以减少 Padding 浪费，提高训练效率。
    
    工作原理：
    Buffer: [样本A][样本B][样本C...]
    如果 Buffer >= max_length，切出前 max_length 个 Token 返回，
    剩余部分保留在 Buffer 中等待下一个样本拼接。
    Args:
        dataset (Dataset): 原始的 SequentialParquetDataset 实例。
        max_length (int): 目标序列长度 (context window)。
        shuffle (bool): 是否打乱原始数据集的读取顺序。默认为 True。
        seed (int): 随机种子。
    '''
    def __init__(self, dataset: SequentialDataset|CachedDataset|ConcatDataset, max_length:int, shuffle=True, seed=42):
        self.dataset = dataset
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.skip_batches = 0 # 待跳过的 Batch 数
    
    def _get_dataset_tokens(self, dataset):
        ''' 递归计算数据集的总 Token 数 '''
        if isinstance(dataset, CachedDataset):
            return dataset.meta.get('total_tokens', 0)
        elif isinstance(dataset, ConcatDataset):
            return sum(self._get_dataset_tokens(ds) for ds in dataset.datasets)
        return 0

    def _get_sample_length(self, dataset, idx):
        ''' 递归获取样本长度 (Helper) '''
        if hasattr(dataset, 'get_sample_length'):
            return dataset.get_sample_length(idx)
        elif isinstance(dataset, ConcatDataset):
            if idx < 0:
                if idx == -1: raise ValueError("Target index has not been reset to 0")
                return 0
            for ds in dataset.datasets:
                l = len(ds)
                if idx < l:
                    return self._get_sample_length(ds, idx)
                idx -= l
        return None

    def __len__(self):
        total_tokens = self._get_dataset_tokens(self.dataset)
        
        if total_tokens > 0:
            return total_tokens // self.max_length
        
        return 0

    def set_epoch(self, epoch):
        ''' 设置当前 Epoch，确保每轮 Shuffle 顺序不同

        Args:
            epoch (int): 当前的 epoch 数。
        '''
        self.epoch = epoch
        
    def set_skip_batches(self, n):
        ''' 设置需要跳过的 Batch 数量 (用于断点续训) '''
        self.skip_batches = n
    
    def __iter__(self):
        ''' 迭代数据集，返回打包后的序列

        在多进程环境下，根据 worker_id 分配数据分片。
        如果启用 shuffle，会在每个 epoch 开始时基于 seed + epoch 打乱索引。
        将变长样本拼接到 buffer 中，当 buffer 长度达到 max_length 时切片返回。

        Yields:
            torch.Tensor: 形状为 (max_length,) 的长整型张量 (input_ids)。
        '''
        worker_info = torch.utils.data.get_worker_info()
        
        num_samples = len(self.dataset)
        indices = list(range(num_samples))
        
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
        
        if worker_info is not None:
            per_worker = int(np.ceil(num_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, num_samples)
            indices = indices[iter_start:iter_end]
            # Resume 在多进程 IterableDataset 下非常困难。
            # 最佳实践是将 num_workers 设为 0 进行 Resume，或者接受数据的一定随机性。
            pass 
            
        buffer_ids = []
        
        skip_tokens = self.skip_batches * self.max_length
        skipped_cnt = 0
        
        for idx in indices:
            # --- Fast Skip Mode ---
            if skip_tokens > 0:
                l = self._get_sample_length(self.dataset, idx)
                
                if l is not None and skipped_cnt + l <= skip_tokens:
                    skipped_cnt += l
                    continue
                
                new_ids = self.dataset[idx]
                if isinstance(new_ids, torch.Tensor): new_ids = new_ids.tolist()
                l = len(new_ids)
                
                if skipped_cnt + l <= skip_tokens:
                    skipped_cnt += l
                    continue
                else:
                    start_offset = skip_tokens - skipped_cnt
                    skip_tokens = 0
                    self.skip_batches = 0
                    
                    if start_offset < l:
                        buffer_ids.extend(new_ids[start_offset:])
                    
                    while len(buffer_ids) >= self.max_length:
                        chunk = buffer_ids[:self.max_length]
                        buffer_ids = buffer_ids[self.max_length:]
                        input_ids = torch.tensor(chunk, dtype=torch.long)
                        yield input_ids
                    
                    continue

            # --- Normal Mode ---
            new_ids = self.dataset[idx]
            
            if len(new_ids) == 0: continue
            
            if isinstance(new_ids, torch.Tensor):
                new_ids = new_ids.tolist()
            
            buffer_ids.extend(new_ids)
            
            while len(buffer_ids) >= self.max_length:
                chunk = buffer_ids[:self.max_length]
                buffer_ids = buffer_ids[self.max_length:]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                
                yield input_ids
