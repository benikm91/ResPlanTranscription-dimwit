package resplan.data

import dimwit.{Tensor, Tensor1, Axis, Label, Random}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py.PyQuote
import org.w3c.dom.Node
import resplan.util.PythonHelper
import dimwit.tensor.VType
import scala.util.Try
import scala.util.Failure
import scala.util.Success

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
  case EndOfNodes extends NodeClass(9, "<eon>")

enum EdgeClass(val id: Int, val name: String):
  case Edge extends EdgeClass(10, "edge")
  case EndOfEdges extends EdgeClass(11, "<eoe>")

trait PlanRenderer:
  def render(plan: RawPlan): Tensor[(Width, Height, Channel), Int]

object StaticRenderer extends PlanRenderer:
  def render(plan: RawPlan): Tensor[(Width, Height, Channel), Int] =
    val img = PythonHelper.pyutil.render_plan_static(plan)
    Tensor.fromPy(VType[Int])(img)

trait GraphLinearization:

  def linearize(nodes: List[RawNode], edges: List[RawEdge]): Tensor1[Vocab, Int]
  def strictUnlinearize(lineraizedGraph: Tensor1[Vocab, Int]): Try[(List[RawNode], List[RawEdge])]

  protected def encodeNode(node: RawNode): NodeClass =
    val nodeClasses = NodeClass.values.toList
    nodeClasses.find(nodeClass => node.nodeName.startsWith(nodeClass.prefix)).head

  protected def encodeEdgeClass(edge: RawEdge): EdgeClass =
    EdgeClass.Edge // for now, we only have one edge class in the dataset

object RandomGraphLinearization extends GraphLinearization:
  def linearize(nodes: List[RawNode], edges: List[RawEdge]): Tensor1[Vocab, Int] =
    val shuffledNodes = scala.util.Random.shuffle(nodes)
    val shuffledEdges = scala.util.Random.shuffle(edges)
    val nodeArray = (shuffledNodes.map(encodeNode).map(_.id) :+ NodeClass.EndOfNodes.id).toArray
    val nodesPos = shuffledNodes.zipWithIndex.map:
      case (node, index) =>
        node.nodeName -> index
    .toMap
    // prepare edge array
    val edgesClass = shuffledEdges.map(encodeEdgeClass)
    val edgesFrom = shuffledEdges.map(edge => nodesPos(edge.fromNodeName))
    val edgesTo = shuffledEdges.map(edge => nodesPos(edge.toNodeName))
    val edgeArray = new Array[Int](shuffledEdges.length * 3 + 1)
    var i = 0
    while i < shuffledEdges.length do
      edgeArray(i * 3 + 0) = edgesClass(i).id
      edgeArray(i * 3 + 1) = edgesFrom(i)
      edgeArray(i * 3 + 2) = edgesTo(i)
      i += 1
    edgeArray(shuffledEdges.length * 3) = EdgeClass.EndOfEdges.id
    Tensor1(Axis[Vocab]).fromArray(nodeArray ++ edgeArray)

  def strictUnlinearize(lineraizedGraph: Tensor1[Vocab, Int]): Try[(List[RawNode], List[RawEdge])] =
    def zipWithClassIndex(list: List[NodeClass]): List[(NodeClass, Int)] =
      var classCounter = Map[NodeClass, Int]().withDefaultValue(0)
      list.map: nodeClass =>
        val idx = classCounter(nodeClass)
        classCounter = classCounter.updated(nodeClass, idx + 1)
        (nodeClass, idx)
    try
      val tokens = py"list(${lineraizedGraph.jaxValue})".as[List[Int]]
      val (nodeTokens, rest) = tokens.splitAt(tokens.indexOf(NodeClass.EndOfNodes.id))
      val edgeTokens = rest.tail
      val nodeClasses = nodeTokens
        .map: nodeId =>
          NodeClass.values.find(_.id == nodeId) match
            case Some(value) => value
            case None        => throw new RuntimeException(f"Unknown node class id: $nodeId")
      val nodes = zipWithClassIndex(nodeClasses).map: (nodeClass, idx) =>
        RawNode(f"${nodeClass.prefix}_$idx")
      assert(edgeTokens.last == EdgeClass.EndOfEdges.id)
      val edges = edgeTokens.init.grouped(3).map: edge =>
        val edgeClass = EdgeClass.values.find(_.id == edge(0)) match
          case Some(value) => value
          case None        => throw new RuntimeException(f"Unknown edge class id: ${edge(0)}")
        val edgeFrom = nodes(edge(1))
        val edgeTo = nodes(edge(2))
        RawEdge(edgeFrom.nodeName, edgeTo.nodeName, edgeClass.name)
      .toList
      Success((nodes, edges))
    catch
      case e: Throwable => Failure(e)

type RawPlan = Map[String, py.Dynamic]
case class RawEdge(fromNodeName: String, toNodeName: String, edgeClassName: String)
case class RawNode(nodeName: String)

case class RawSample(plan: RawPlan, nodes: List[RawNode], edges: List[RawEdge])
case class Sample(image: Tensor[(Width, Height, Channel), Int], lineraizedGraph: Tensor1[Vocab, Int])

object ResPlanDataset:
  // 1. Load data via ScalaPy
  private val pickle = py.module("pickle")
  private val builtins = py.module("builtins")

  def loadRawPlans(dataPath: String = "res/ResPlan.pkl"): List[RawSample] =
    val file = builtins.open(dataPath, "rb")
    val data =
      try
        pickle.load(file)
      catch
        case e: Throwable =>
          throw RuntimeException(s"Error loading data from $dataPath. Maybe forgot to run 'git lfs pull'? Original error: ${e.getMessage}")
      finally
        file.close()
    data.as[List[Map[String, py.Dynamic]]]
      .map: plan =>
        // fix "balacony" => "balcony" typo in the dataset
        if plan.contains("balacony")
        then plan.updated("balcony", plan("balacony")).removed("balacony")
        else plan
      .map: plan =>
        val graph = plan("graph")
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
