package resplan.data

import dimwit.{Tensor, Tensor1, Axis, Label, Random}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import org.w3c.dom.Node
import resplan.util.PythonHelper
import dimwit.tensor.VType

trait Width derives Label
trait Height derives Label
trait Channel derives Label
trait Node derives Label
trait Edge derives Label

enum NodeClass(val id: Int, val prefix: String):
  case Living extends NodeClass(0, "living")
  case Bedroom extends NodeClass(1, "bedroom")
  case Bathroom extends NodeClass(2, "bathroom")
  case Kitchen extends NodeClass(3, "kitchen")
  case Door extends NodeClass(4, "door")
  case Window extends NodeClass(5, "window")
  case Wall extends NodeClass(6, "wall")
  case FrontDoor extends NodeClass(7, "front_door")
  case Balcony extends NodeClass(8, "balcony")
  case EOS extends NodeClass(9, "<eos>")

enum EdgeClass(val id: Int):
  case Edge extends EdgeClass(0)
  case EOS extends EdgeClass(1)

trait PlanRenderer:
  def render(plan: RawPlan): Tensor[(Width, Height, Channel), Int]

object StaticRenderer extends PlanRenderer:
  def render(plan: RawPlan): Tensor[(Width, Height, Channel), Int] =
    val img = PythonHelper.pyutil.render_plan_static(plan)
    Tensor.fromPy(VType[Int])(img)

case class LinearizedGraph(
    nodes: Tensor1[Node, Int],
    edgesFrom: Tensor1[Edge, Int],
    edgesTo: Tensor1[Edge, Int],
    edgesClass: Tensor1[Edge, Int]
)

trait GraphLinearization:
  def encodeNode(node: RawNode): Int =
    val nodeClasses = NodeClass.values.toList
    nodeClasses.find(nodeClass => node.nodeName.startsWith(nodeClass.prefix)).head.id
  def encodeEdgeClass(edge: RawEdge): Int =
    // for now, we only have one edge class in the dataset
    EdgeClass.Edge.id
  def linearize(nodes: List[RawNode], edges: List[RawEdge]): LinearizedGraph

object RandomGraphLinearization extends GraphLinearization:
  def linearize(nodes: List[RawNode], edges: List[RawEdge]): LinearizedGraph =
    val nodeIds = nodes.map(encodeNode)
    val cup = nodes.zip(nodeIds)
    val shuffledCup = scala.util.Random.shuffle(cup)
    val (shuffledNodes, shuffledNodeIds) = shuffledCup.unzip
    val nodesPos = nodes.map(_.nodeName).zipWithIndex.toMap
    val edgesFrom = edges.map(edge => nodesPos(edge.fromNodeName))
    val edgesTo = edges.map(edge => nodesPos(edge.toNodeName))
    val edgesClass = edges.map(encodeEdgeClass)
    LinearizedGraph(
      Tensor1(Axis[Node]).fromArray(shuffledNodeIds.toArray),
      Tensor1(Axis[Edge]).fromArray(edgesFrom.toArray),
      Tensor1(Axis[Edge]).fromArray(edgesTo.toArray),
      Tensor1(Axis[Edge]).fromArray(edgesClass.toArray)
    )

type RawPlan = Map[String, py.Dynamic]
case class RawEdge(fromNodeName: String, toNodeName: String, edgeClassName: String)
case class RawNode(nodeName: String)

case class RawSample(plan: RawPlan, nodes: List[RawNode], edges: List[RawEdge])
case class Sample(image: Tensor[(Width, Height, Channel), Int], graph: LinearizedGraph)

object ResPlanDataset:
  // 1. Load data via ScalaPy
  private val pickle = py.module("pickle")
  private val builtins = py.module("builtins")

  def loadRawPlans(dataPath: String): List[RawSample] =
    val file = builtins.open(dataPath, "rb")
    val data = pickle.load(file)
    file.close()
    data.as[List[Map[String, py.Dynamic]]]
      .map: plan =>
        // fix "balacony" => "balcony" typo in the dataset
        if plan.contains("balacony")
        then plan.updated("balcony", plan("balacony")).removed("balacony")
        else plan
      .map: plan =>
        val graph = plan("graph")
        import me.shadaj.scalapy.py.PyQuote
        val nodes = py"list(${graph.nodes})".as[List[String]].map(RawNode(_))
        val edges = py"list(${graph.edges(data = true)})".as[List[(String, String, py.Dynamic)]].map: (from, to, data) =>
          RawEdge(from, to, data.bracketAccess("type").as[String])
        RawSample(plan, nodes, edges)

class ResPlanDataset(
    plans: List[RawSample],
    graphLinearization: GraphLinearization,
    renderer: PlanRenderer
):
  def length: Int = plans.length
  def iterator: Iterator[Sample] =
    plans.iterator.map(rawSample =>
      val linearizedGraph = graphLinearization.linearize(rawSample.nodes, rawSample.edges)
      Sample(renderer.render(rawSample.plan), linearizedGraph)
    )
