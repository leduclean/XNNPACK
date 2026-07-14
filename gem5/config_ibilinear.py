import argparse
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

p = argparse.ArgumentParser()
p.add_argument("kernel")
p.add_argument("--binary", default="bench_ibilinear")
args = p.parse_args()

cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="32KiB", l1i_size="32KiB", l2_size="256KiB"
)

memory = SingleChannelDDR3_1600("2GiB")

processor = SimpleProcessor(cpu_type=CPUTypes.O3, num_cores=1, isa=ISA.RISCV)

board = SimpleBoard(
    clk_freq="1GHz", processor=processor, memory=memory, cache_hierarchy=cache_hierarchy
)


board.set_se_binary_workload(BinaryResource(args.binary), arguments=[args.kernel])
simulator = Simulator(board=board)
simulator.run()
