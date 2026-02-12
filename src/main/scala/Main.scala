package resplan

import dimwit.*
import resplan.data.ResPlanDataset
import resplan.data.RandomGraphLinearization
import resplan.data.StaticRenderer
import resplan.util.PythonHelper
import resplan.data.RawNode
import resplan.data.RawEdge
import java.io.PrintWriter

@main def dataExample(): Unit =
  val plans = ResPlanDataset.loadRawPlans()
  val dataset = ResPlanDataset(
    plans,
    RandomGraphLinearization,
    StaticRenderer
  )
  println(dataset.length)

  val sample = dataset.iterator.next

  println("Original graph:")
  val origNodes = plans.head.nodes
  val origEdges = plans.head.edges
  writeQuickDot(quickDot(origNodes, origEdges), "orig.dot")
  println(origNodes.map(_.nodeName))
  println(origEdges.map(edge => f"${edge.fromNodeName} <-> ${edge.toNodeName}"))

  println("Linearizing graph...")
  println(sample.lineraizedGraph)

  println("Unlinearizing graph...")
  val (nodes, edges) = RandomGraphLinearization.strictUnlinearize(sample.lineraizedGraph).get
  println(nodes.map(_.nodeName))
  println(edges.map(edge => f"${edge.fromNodeName} <-> ${edge.toNodeName}"))

  writeQuickDot(quickDot(nodes, edges), "unlineraized.dot")

  val plt = PythonHelper.pyplot
  plt.imshow(sample.image.jaxValue)
  plt.show()

def writeQuickDot(dot: String, filename: String): Unit =
  new PrintWriter(filename):
    write(dot)
    close()

def quickDot(nodes: List[RawNode], edges: List[RawEdge]): String =
  s"""graph G {
     |  layout=neato; overlap=false;
     |  ${nodes.map(n => s""""${n.nodeName}";""").mkString("\n  ")}
     |  ${edges.map(e => s""""${e.fromNodeName}" -- "${e.toNodeName}" [label="${e.edgeClassName}"];""").mkString("\n  ")}
     |}""".stripMargin
