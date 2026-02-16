package resplan

import dimwit.*
import dimwit.Conversions.given
import resplan.data.ResPlanDataset
import resplan.data.RandomGraphLinearization
import resplan.util.PythonHelper
import resplan.data.RawNode
import resplan.data.RawEdge
import java.io.PrintWriter
import resplan.data.Graph
import me.shadaj.scalapy.py.SeqConverters
import resplan.data.{Data, Width, Height, Channel}
import dimwit.jax.Jax
import resplan.data.ImageAugmentation
import resplan.data.RandomUtil.toSourceOfRandomness
import scala.concurrent.ExecutionContext.Implicits.global

@main def dataExample(): Unit =
  val plans = ResPlanDataset.loadRawPlans()
  val planImages = Tensor.fromPy[(Data, Width, Height, Channel), Int](VType[Int])(
    Jax.jnp.asarray(PythonHelper.np.memmap("res/plans/20/plan_imgs.bin", dtype = PythonHelper.np.uint8, shape = (17107, 160, 160, 3), mode = "r"))
  )
  val dataset = ResPlanDataset(
    plans,
    planImages,
    RandomGraphLinearization(),
    image =>
      image.asFloat /! 255f,
    Random.Key(42).toSourceOfRandomness
  )
  println(dataset.length)

  val sample = dataset.iterator.next

  println("Original graph:")
  val origGraph = plans.head.graph
  writeQuickDot(quickDot(origGraph), "orig.dot")
  println(origGraph.nodes.map(_.nodeName))
  println(origGraph.edges.map(edge => f"${edge.fromNodeName} <-> ${edge.toNodeName}"))

  println("Linearizing graph...")
  println(sample.lineraizedGraph)

  println("Unlinearizing graph...")
  val graph = RandomGraphLinearization().strictUnlinearize(sample.lineraizedGraph).get
  println(graph.nodes.map(_.nodeName))
  println(graph.edges.map(edge => f"${edge.fromNodeName} <-> ${edge.toNodeName}"))

  writeQuickDot(quickDot(graph), "unlineraized.dot")

  val plt = PythonHelper.pyplot

  val numRows = 10
  val numCols = 10
  val totalAugs = numRows * numCols

  plt.figure(figsize = (15, 15))
  plt.subplot(numRows, numCols, 1)
  plt.imshow(sample.image.jaxValue)
  for (i, augKey) <- (2 to totalAugs).zip(Random.Key(42).toSourceOfRandomness) do
    val augImage = ImageAugmentation(sample.image, augKey)
    plt.subplot(numRows, numCols, i)
    plt.imshow(augImage.jaxValue)
  plt.tight_layout()
  plt.show()

def writeQuickDot(dot: String, filename: String): Unit =
  new PrintWriter(filename):
    write(dot)
    close()

def quickDot(graph: Graph): String =
  s"""graph G {
     |  layout=neato; overlap=false;
     |  ${graph.nodes.map(n => s""""${n.nodeName}";""").mkString("\n  ")}
     |  ${graph.edges.map(e => s""""${e.fromNodeName}" -- "${e.toNodeName}" [label="${e.edgeClassName}"];""").mkString("\n  ")}
     |}""".stripMargin
