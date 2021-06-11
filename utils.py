# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
import io
import os
import time
import subprocess
import socket
from collections import defaultdict, deque
import datetime
from logging import getLogger

import torch
import torch.distributed as dist

logger = getLogger()

# class SmoothedValue(object):
#     """Track a series of values and provide access to smoothed values over a
#     window or the global series average.
#     """

#     def __init__(self, window_size=20, fmt=None):
#         if fmt is None:
#             fmt = "{median:.4f} ({global_avg:.4f})"
#         self.deque = deque(maxlen=window_size)
#         self.total = 0.0
#         self.count = 0
#         self.fmt = fmt

#     def update(self, value, n=1):
#         self.deque.append(value)
#         self.count += n
#         self.total += value * n

#     def synchronize_between_processes(self):
#         """
#         Warning: does not synchronize the deque!
#         """
#         if not is_dist_avail_and_initialized():
#             return
#         t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
#         dist.barrier()
#         dist.all_reduce(t)
#         t = t.tolist()
#         self.count = int(t[0])
#         self.total = t[1]

#     @property
#     def median(self):
#         d = torch.tensor(list(self.deque))
#         return d.median().item()

#     @property
#     def avg(self):
#         d = torch.tensor(list(self.deque), dtype=torch.float32)
#         return d.mean().item()

#     @property
#     def global_avg(self):
#         return self.total / self.count

#     @property
#     def max(self):
#         return max(self.deque)

#     @property
#     def value(self):
#         return self.deque[-1]

#     def __str__(self):
#         return self.fmt.format(
#             median=self.median,
#             avg=self.avg,
#             global_avg=self.global_avg,
#             max=self.max,
#             value=self.value)


# class MetricLogger(object):
#     def __init__(self, delimiter="\t"):
#         self.meters = defaultdict(SmoothedValue)
#         self.delimiter = delimiter

#     def update(self, **kwargs):
#         for k, v in kwargs.items():
#             if isinstance(v, torch.Tensor):
#                 v = v.item()
#             assert isinstance(v, (float, int))
#             self.meters[k].update(v)

#     def __getattr__(self, attr):
#         if attr in self.meters:
#             return self.meters[attr]
#         if attr in self.__dict__:
#             return self.__dict__[attr]
#         raise AttributeError("'{}' object has no attribute '{}'".format(
#             type(self).__name__, attr))

#     def __str__(self):
#         loss_str = []
#         for name, meter in self.meters.items():
#             loss_str.append(
#                 "{}: {}".format(name, str(meter))
#             )
#         return self.delimiter.join(loss_str)

#     def synchronize_between_processes(self):
#         for meter in self.meters.values():
#             meter.synchronize_between_processes()

#     def add_meter(self, name, meter):
#         self.meters[name] = meter

#     def log_every(self, iterable, print_freq, header=None):
#         i = 0
#         if not header:
#             header = ''
#         start_time = time.time()
#         end = time.time()
#         iter_time = SmoothedValue(fmt='{avg:.4f}')
#         data_time = SmoothedValue(fmt='{avg:.4f}')
#         space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
#         log_msg = [
#             header,
#             '[{0' + space_fmt + '}/{1}]',
#             'eta: {eta}',
#             '{meters}',
#             'time: {time}',
#             'data: {data}'
#         ]
#         if torch.cuda.is_available():
#             log_msg.append('max mem: {memory:.0f}')
#         log_msg = self.delimiter.join(log_msg)
#         MB = 1024.0 * 1024.0
#         for obj in iterable:
#             data_time.update(time.time() - end)
#             yield obj
#             iter_time.update(time.time() - end)
#             if i % print_freq == 0 or i == len(iterable) - 1:
#                 eta_seconds = iter_time.global_avg * (len(iterable) - i)
#                 eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
#                 if torch.cuda.is_available():
#                     print(log_msg.format(
#                         i, len(iterable), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time),
#                         memory=torch.cuda.max_memory_allocated() / MB))
#                 else:
#                     print(log_msg.format(
#                         i, len(iterable), eta=eta_string,
#                         meters=str(self),
#                         time=str(iter_time), data=str(data_time)))
#             i += 1
#             end = time.time()
#         total_time = time.time() - start_time
#         total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#         print('{} Total time: {} ({:.4f} s / it)'.format(
#             header, total_time_str, total_time / len(iterable)))


# def _load_checkpoint_for_ema(model_ema, checkpoint):
#     """
#     Workaround for ModelEma._load_checkpoint to accept an already-loaded object
#     """
#     mem_file = io.BytesIO()
#     torch.save(checkpoint, mem_file)
#     mem_file.seek(0)
#     model_ema._load_checkpoint(mem_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ and not params.debug_slurm
    logger.info("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    if params.is_slurm_job:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            logger.info(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        assert 10001 <= params.master_port <= 20000 or params.world_size == 1
        logger.info(PREFIX + "Master address: %s" % params.master_addr)
        logger.info(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.master_addr
        os.environ['MASTER_PORT'] = str(params.master_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif params.local_rank != -1:

        assert params.master_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['NGPU'])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    # local job (single GPU)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.distributed = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "Global rank    : %i" % params.global_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.distributed))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.distributed:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        logger.info("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )


# def init_distributed_mode(args):
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = args.rank % torch.cuda.device_count()
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)