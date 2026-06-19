from .lasot import Lasot
from .got10k import Got10k
from .tracking_net import TrackingNet
from .imagenetvid import ImagenetVID
try:
    from .coco import MSCOCO
    from .coco_seq import MSCOCOSeq
except ImportError:
    MSCOCO = None
    MSCOCOSeq = None
from .got10k_lmdb import Got10k_lmdb
from .lasot_lmdb import Lasot_lmdb
from .imagenetvid_lmdb import ImagenetVID_lmdb
try:
    from .coco_seq_lmdb import MSCOCOSeq_lmdb
except ImportError:
    MSCOCOSeq_lmdb = None
from .tracking_net_lmdb import TrackingNet_lmdb
from .aic_contest import AICContest
