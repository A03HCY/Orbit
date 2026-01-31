from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING: from orbit.engine import Engine

def pre_train(engine: Engine):
    outputs = engine.model(engine.data)
    logits = outputs.logits
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = engine.data[..., 1:].contiguous()

    loss = engine.criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    engine.update(loss)