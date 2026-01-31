import json
import random
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import tqdm
import os
import concurrent.futures
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer


class SequentialParquetDataset(Dataset):
    ''' 顺序读取Parquet文件的数据集

    该数据集扫描指定目录下的所有Parquet文件，建立索引，并提供基于索引的随机访问。
    支持自定义列映射配置，以适应不同的数据格式。

    Args:
        root_dir (str | Path): 包含Parquet文件的数据集根目录。
        tokenizer (str | PreTrainedTokenizer): 用于编码文本的分词器或分词器路径。
        config (dict, optional): 文件夹路径关键字到列名映射的配置字典。默认为 None。
        fim_rate (float): 启用中间填充(Fill-In-the-Middle)任务的概率，默认为 0.5。
        max_length (int, optional): 序列最大长度。默认为 None。
        in_memory (bool): 是否将所有数据预先加载到内存中。默认为 False。
                          开启后启动时间变长，占用大量 RAM，但训练时无磁盘 IO。
    '''
    def __init__(self, root_dir, tokenizer, config=None, fim_rate=0.5, max_length=None, in_memory=False):
        self.root_dir = Path(root_dir)
        
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
            
        self.config = config or {}
        self.fim_rate = fim_rate
        self.max_length = max_length
        self.in_memory = in_memory
        
        self.data_entries = []
        self.total_rows = 0
        self._build_index()

    def _get_mapping_for_path(self, file_path):
        ''' 根据文件路径获取对应的列名映射 '''
        path_str = str(file_path).replace('\\', '/')
        for folder_key, mapping in self.config.items():
            if folder_key in path_str:
                return mapping
        
        return {
            'system': 'system',           # 系统提示词 / System Prompt
            'train': 'text',              # 预训练文本 / Pretrain Data
            'user': 'user',               # 用户输入 / User Instruction
            'model': 'model',             # 模型文本回复 / Answer
            'reasoning': 'thought',       # 思维链 / CoT
            'tool_calls': 'tool_calls',   # 模型发起的工具调用
            'tool_result': 'tool_output'  # 工具返回的结果
        }

    def _build_index(self):
        ''' 扫描目录建立文件索引，可选加载数据到内存 '''
        print(f'[SequentialParquetDataset] Scanning {self.root_dir} (In-Memory: {self.in_memory})...')
        files = list(self.root_dir.rglob('*.parquet'))
        
        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc="Indexing")
        except ImportError:
            iterator = files

        for file_path in iterator:
            mapping = self._get_mapping_for_path(file_path)
            
            cached_df = None
            if self.in_memory:
                try:
                    cols_to_load = [v for v in mapping.values() if v]
                    if cols_to_load:
                        cached_df = pd.read_parquet(file_path, columns=cols_to_load, engine='pyarrow')
                        num_rows = len(cached_df)
                    else:
                        num_rows = 0
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
            else:
                # 2. 磁盘模式：只读元数据
                try:
                    meta = pq.read_metadata(file_path)
                    num_rows = meta.num_rows
                except Exception as e:
                    print(f"Error reading metadata {file_path}: {e}")
                    continue

            if num_rows == 0:
                continue

            self.data_entries.append({
                'path': file_path,
                'start_idx': self.total_rows,
                'end_idx': self.total_rows + num_rows,
                'mapping': mapping,
                'cache': cached_df
            })
            self.total_rows += num_rows
            
        print(f'[SequentialParquetDataset] Indexed {len(self.data_entries)} files with {self.total_rows} total rows.')

    def __len__(self):
        return self.total_rows

    def _create_fim_dict(self, text):
        ''' 创建 FIM (Fill-In-the-Middle) 任务样本 '''
        if len(text) < 50: return None
        total_len = len(text)
        span_len = int(total_len * np.random.uniform(0.1, 0.3))
        if span_len < 5: return None
        start = np.random.randint(0, total_len - span_len)
        return {
            'role': 'fim', 'prefix': text[:start], 'middle': text[start:start+span_len], 'suffix': text[start+span_len:]
        }

    def _safe_json_load(self, content):
        ''' 安全解析 JSON 内容 '''
        if not content: return None
        if isinstance(content, list) or isinstance(content, dict):
            return content
        try:
            return json.loads(str(content))
        except:
            return None

    def __getitem__(self, idx):
        ''' 获取指定索引的数据样本 '''
        entry = next(e for e in self.data_entries if e['start_idx'] <= idx < e['end_idx'])
        mapping = entry['mapping']
        local_idx = idx - entry['start_idx']
        
        if self.in_memory and entry['cache'] is not None:
            df_row = entry['cache'].iloc[local_idx]
        else:
            cols = [v for v in mapping.values() if v]
            df_row = pd.read_parquet(
                entry['path'], 
                columns=cols, 
                engine='pyarrow'
            ).iloc[local_idx]

        def get(key):
            col_name = mapping.get(key)
            
            if col_name and col_name in df_row.index:
                val = df_row[col_name]
                if pd.notna(val):
                    s = val
                    if isinstance(s, str):
                        s = s.strip()
                        return s if len(s) > 0 else None
                    return s
            return None

        messages = []

        sys_text = get('system')
        if sys_text:
            messages.append({'role': 'system', 'content': str(sys_text)})

        train_text = get('train')
        if train_text:
            train_text = str(train_text)
            if self.fim_rate > 0 and np.random.rand() < self.fim_rate:
                fim_msg = self._create_fim_dict(train_text)
                messages.append(fim_msg if fim_msg else {'role': 'train', 'content': train_text})
            else:
                messages.append({'role': 'train', 'content': train_text})

        user_text = get('user')
        if user_text:
            messages.append({'role': 'user', 'content': str(user_text)})

        model_text = get('model')
        reasoning_text = get('reasoning')
        tool_calls_raw = get('tool_calls')
        
        if model_text or reasoning_text or tool_calls_raw:
            msg = {
                'role': 'model',
                'content': str(model_text) if model_text else '',
                'reasoning_content': str(reasoning_text) if reasoning_text else None
            }
            
            if tool_calls_raw:
                parsed_tools = self._safe_json_load(tool_calls_raw)
                if parsed_tools:
                    msg['tool_calls'] = parsed_tools
            
            messages.append(msg)

        tool_res = get('tool_result')
        if tool_res:
             messages.append({'role': 'tool', 'content': str(tool_res)})

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
        
        ids = tokenized[0]
        return ids


class CachedDataset(Dataset):
    ''' 缓存加速数据集 (Pre-tokenized Cache)
    '''
    def __init__(
        self, 
        source_dataset, 
        cache_dir, 
        max_length=2048, 
        chunk_size=1024*1024*1024*2,
        rebuild=False, 
        num_workers=16
    ):
        self.source_dataset = source_dataset
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.num_workers = num_workers
        
        self.idx_path = os.path.join(cache_dir, "index.bin")
        self.meta_path = os.path.join(cache_dir, "meta.json")

        if rebuild or not self._check_exists():
            self._build_cache()
        
        self._load_cache()

    def _check_exists(self):
        # 只要 index 和 meta 存在，且 data_0.bin 存在即可
        return os.path.exists(self.idx_path) and \
               os.path.exists(self.meta_path) and \
               os.path.exists(os.path.join(self.cache_dir, "data_0.bin"))

    def _process_sample(self, idx):
        """ 子线程任务：读取并 Tokenize """
        try:
            raw_sample = self.source_dataset[idx]
            
            # 自动探测数据格式
            if hasattr(raw_sample, 'ids'):
                ids = raw_sample.ids
            elif isinstance(raw_sample, dict) and 'input_ids' in raw_sample:
                ids = raw_sample['input_ids']
            else:
                ids = raw_sample

            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            
            if not ids or len(ids) == 0:
                return None

            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            
            # 转为 bytes
            np_ids = np.array(ids, dtype=self.dtype)
            return np_ids.tobytes()
        except Exception as e:
            return f"Error: {str(e)}"

    def _build_cache(self):
        ''' 构建缓存：调试模式 '''
        import sys
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        total_samples = len(self.source_dataset)
        print(f"[Debug] Checking Source Dataset...")
        print(f"   - Length: {total_samples}")
        
        if total_samples == 0:
            raise ValueError("Error: Source dataset is empty (len=0). Check your root_dir path!")

        print("   - Testing reading sample [0]...")
        try:
            test_sample = self.source_dataset[0] 
            print(f"     -> Success. Type: {type(test_sample)}")
            if isinstance(test_sample, dict):
                print(f"     -> Keys: {test_sample.keys()}")
                if 'input_ids' in test_sample:
                    print(f"     -> Input IDs shape: {test_sample['input_ids'].shape}")
            elif isinstance(test_sample, torch.Tensor):
                print(f"     -> Tensor shape: {test_sample.shape}")
        except Exception as e:
            print(f"Error reading sample [0]: {e}")
            raise e

        vocab_size = getattr(getattr(self.source_dataset, 'tokenizer', None), "vocab_size", 65536)
        self.dtype = np.uint16 if vocab_size < 65535 else np.uint32
        
        indices = [] 
        current_file_id = 0
        current_file_ptr = 0 
        total_tokens = 0
        
        f_current = open(os.path.join(self.cache_dir, f"data_{current_file_id}.bin"), "wb")
        
        print(f"[Debug] Start Caching loop (Workers={self.num_workers})...")
        sys.stdout.flush()

        try:
            if self.num_workers <= 1:
                print("Running in MAIN THREAD (No Executor). Errors will crash the script.")
                
                for idx in tqdm.tqdm(range(total_samples), desc="Caching (Serial)"):
                    
                    raw_sample = self.source_dataset[idx]
                    
                    if hasattr(raw_sample, 'ids'): ids = raw_sample.ids
                    elif isinstance(raw_sample, dict) and 'input_ids' in raw_sample: ids = raw_sample['input_ids']
                    else: ids = raw_sample

                    if isinstance(ids, torch.Tensor): ids = ids.tolist()
                    
                    if not ids or len(ids) == 0: continue

                    if len(ids) > self.max_length: ids = ids[:self.max_length]
                    
                    np_ids = np.array(ids, dtype=self.dtype)
                    res = np_ids.tobytes()
                    
                    byte_len = len(res)
                    item_len = byte_len // np.dtype(self.dtype).itemsize
                    
                    if (current_file_ptr + byte_len > self.chunk_size) and (current_file_ptr > 0):
                        f_current.close()
                        current_file_id += 1
                        current_file_ptr = 0
                        f_current = open(os.path.join(self.cache_dir, f"data_{current_file_id}.bin"), "wb")
                    
                    f_current.write(res)
                    indices.append((current_file_id, current_file_ptr, item_len))
                    current_file_ptr += byte_len
                    total_tokens += item_len

            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    results = executor.map(self._process_sample, range(total_samples), chunksize=1)
                    
                    iterator = tqdm.tqdm(results, total=total_samples, desc="Caching (Threaded)", ncols=100)
                    
                    for res in iterator:
                        if res is None: continue
                        
                        if isinstance(res, str) and res.startswith("Error"):
                            print(f"\n{res}")
                            continue
                        
                        byte_len = len(res)
                        item_len = byte_len // np.dtype(self.dtype).itemsize
                        
                        if (current_file_ptr + byte_len > self.chunk_size) and (current_file_ptr > 0):
                            f_current.close()
                            current_file_id += 1
                            current_file_ptr = 0
                            f_current = open(os.path.join(self.cache_dir, f"data_{current_file_id}.bin"), "wb")
                        
                        f_current.write(res)
                        indices.append((current_file_id, current_file_ptr, item_len))
                        current_file_ptr += byte_len
                        total_tokens += item_len

        finally:
            f_current.close()

        print("\nSaving index file...")
        np_indices = np.array(indices, dtype=np.uint64)
        np.save(self.idx_path, np_indices)
        
        meta = {
            "dtype": np.dtype(self.dtype).name,
            "total_samples": len(indices),
            "total_tokens": int(total_tokens),
            "max_length": self.max_length,
            "num_chunks": current_file_id + 1
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
            
        print(f"Cache Done! Tokens: {total_tokens}")



    def _load_cache(self):
        ''' 加载缓存：建立多个 memmap '''
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
            
        self.dtype = np.dtype(self.meta["dtype"])
        self.num_chunks = self.meta.get("num_chunks", 1) 
        
        self.indices = np.load(self.idx_path, mmap_mode='r')
        self.num_samples = len(self.indices)
        
        self.mmaps = []
        for i in range(self.num_chunks):
            path = os.path.join(self.cache_dir, f"data_{i}.bin")
            
            if os.path.exists(path):
                m = np.memmap(path, dtype=self.dtype, mode='r')
                self.mmaps.append(m)
            else:
                raise FileNotFoundError(f"Cache chunk missing: {path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_id, offset, length = self.indices[idx]
        
        item_offset = int(offset // np.dtype(self.dtype).itemsize)
        
        mmap = self.mmaps[int(file_id)]
        np_array = np.array(mmap[item_offset : item_offset + int(length)], copy=True)
        
        return torch.from_numpy(np_array.astype(np.int64))


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
    def __init__(self, dataset: SequentialParquetDataset|CachedDataset, max_length:int, shuffle=True, seed=42):
        self.dataset = dataset
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
    
    def set_epoch(self, epoch):
        ''' 设置当前 Epoch，确保每轮 Shuffle 顺序不同

        Args:
            epoch (int): 当前的 epoch 数。
        '''
        self.epoch = epoch
    
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
            
        buffer_ids = []
        
        for idx in indices:
            new_ids = self.dataset[idx].ids
            
            if len(new_ids) == 0: continue
            
            buffer_ids.extend(new_ids)
            
            while len(buffer_ids) >= self.max_length:
                chunk = buffer_ids[:self.max_length]
                buffer_ids = buffer_ids[self.max_length:]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                
                yield input_ids

