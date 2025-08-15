# type: ignore

from src.machine_learning import run_and_MLclassify_TNO
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput


config = Config()
config.trace_filter = GlobbingFilter(
    include=[
        "src.*",
    ]
)

graphviz = GraphvizOutput(output_file="call_graph/pycallgraph.png")

with PyCallGraph(output=graphviz, config=config):
    flag, tno_class, sim = run_and_MLclassify_TNO(
        des="K16F59G",
        datadir="proto/call_graph",
        logfile="screen",
        deletefile=True,
    )
    tno_class.print_results()
