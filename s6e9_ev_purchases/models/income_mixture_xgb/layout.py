"""Cache and run folders of this model, and the caches it reads from other model folders.

cache/stage1   base tables, nested target encodings, nested LR margins, exact-income EM, fold matrices
cache/stage2   income context teachers and context-shape artifacts
cache/stage8   income digit features
cache/em_info  information-aware income EM pipeline (base, teachers, EM, information, Bayes margins)
runs/<stage>/<run>/predictions.npz   per fold / seed predictions (index = held-out fold rows, then all test rows)"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent
CACHE = HERE / 'cache'
STAGE1 = CACHE / 'stage1'
STAGE2 = CACHE / 'stage2'
STAGE8 = CACHE / 'stage8'
EM_INFO = CACHE / 'em_info'
RUNS = HERE / 'runs'

# caches of other folders (original file names kept below each folder)
NEURAL = MODELS / 'neural_contrasts' / 'cache'      # stage3 ... stage6
RESPONSE = MODELS / 'response_features' / 'cache'   # base, static, bpe
