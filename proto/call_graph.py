# type: ignore

from pycallgraph import Config, GlobbingFilter, PyCallGraph
from pycallgraph.output import GraphvizOutput

from sbdynt.machine_learning import run_and_MLclassify_TNO

config = Config()
config.trace_filter = GlobbingFilter(
    include=[
        "sbdynt.*",
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
