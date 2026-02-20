package resplan.data

import dimwit.*
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py.PyQuote
import org.w3c.dom.Node
import resplan.util.PythonHelper
import dimwit.tensor.VType
import scala.util.Try
import scala.util.Failure
import scala.util.Success
import dimwit.tensor.Tensor3
import dimwit.tensor.Shape
import RandomUtil.toSourceOfRandomness
import scala.concurrent.Future
import scala.concurrent.ExecutionContext
import dimwit.tensor.DeviceBackend.GPU

trait Data derives Label
trait Width derives Label
trait Height derives Label
trait Channel derives Label
trait Node derives Label
trait Edge derives Label

val paddingValue = -1

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

object NodeClass:
  def fromName(nodeName: String): NodeClass =
    val nodeClasses = NodeClass.values.toList
    nodeClasses.find(nodeClass => nodeName.startsWith(nodeClass.prefix)).head

enum EdgeClass(val id: Int, val name: String):
  case Edge extends EdgeClass(10, "edge")
  case EndOfEdges extends EdgeClass(11, "<eoe>")

val BOS = 12

val POS_OFFSET = 13

def isIsomorphicTo(pred: Graph, target: Graph): Boolean =
  if pred.nodes.size != target.nodes.size || pred.edges.size != target.edges.size
  then false
  else
    val predNodes = pred.nodes.toArray
    val targetNodes = target.nodes.toArray
    val pAdj = pred.edges.groupBy(_.fromNode.nodeName)
    val tAdj = target.edges.groupBy(_.fromNode.nodeName)

    def isValid(pName: String, tName: String, mapping: Map[String, String]): Boolean =
      val pOut = pAdj.getOrElse(pName, Nil)
      val tOut = tAdj.getOrElse(tName, Nil)

      pOut.size == tOut.size && pOut.forall: pe =>
        mapping.get(pe.toNode.nodeName).forall: mDest =>
          tOut.exists(te => te.edgeClassName == pe.edgeClassName && te.toNode.nodeName == mDest)

    def solve(idx: Int, used: Set[Int], mapping: Map[String, String]): Boolean =
      if idx == predNodes.length
      then return true
      else
        val pNode = predNodes(idx)
        targetNodes.indices.filterNot(used).exists: tIdx =>
          val tNode = targetNodes(tIdx)
          pNode.nodeClass == tNode.nodeClass &&
          isValid(pNode.nodeName, tNode.nodeName, mapping + (pNode.nodeName -> tNode.nodeName)) &&
          solve(idx + 1, used + tIdx, mapping + (pNode.nodeName -> tNode.nodeName))

    solve(0, Set.empty, Map.empty)

def areNodesCorrect(predNodes: List[RawNode], targetNodes: List[RawNode]): Boolean =
  predNodes.size == targetNodes.size && predNodes.diff(targetNodes).isEmpty

case class GraphLinearizationScore(structureCorrect: Boolean, nodesCorrect: Boolean, graphCorrect: Boolean)

extension (b: Boolean)
  def toFloat: Float = if b then 1.0f else 0.0f

trait GraphLinearization:

  def linearize(graph: Graph): Tensor1[DecoderContext, Int]
  def strictUnlinearize(lineraizedGraph: Tensor1[DecoderContext, Int]): Try[Graph]
  def score(pred: Tensor1[DecoderContext, Int], target: Tensor1[DecoderContext, Int]): GraphLinearizationScore =
    strictUnlinearize(pred) match
      case Failure(exception) =>
        GraphLinearizationScore(false, false, false)
      case Success(value) =>
        val targetGraph = strictUnlinearize(target).get
        GraphLinearizationScore(true, areNodesCorrect(value.nodes, targetGraph.nodes), isIsomorphicTo(value, targetGraph))

// scala.util.Random.shuffle(nodes)
trait NodeLineraization:

  def encodeNode(node: RawNode): NodeClass =
    NodeClass.fromName(node.nodeName)

  def shuffleNodes(nodes: List[RawNode]): List[RawNode]

  def toNodeArray(nodes: List[RawNode]): Array[Int] =
    (nodes.map(encodeNode).map(_.id) :+ NodeClass.EndOfNodes.id).toArray

  def toNodes(nodeTokens: List[Int]): List[RawNode] =
    def zipWithClassIndex(list: List[NodeClass]): List[(NodeClass, Int)] =
      var classCounter = Map[NodeClass, Int]().withDefaultValue(0)
      list.map: nodeClass =>
        val idx = classCounter(nodeClass)
        classCounter = classCounter.updated(nodeClass, idx + 1)
        (nodeClass, idx)
    val nodeClasses = nodeTokens
      .map: nodeId =>
        NodeClass.values.find(_.id == nodeId) match
          case Some(value) => value
          case None        => throw new RuntimeException(f"Unknown node class id: $nodeId")
    zipWithClassIndex(nodeClasses).map: (nodeClass, idx) =>
      RawNode(f"${nodeClass.prefix}_$idx", nodeClass)

trait EdgeLineraization:

  protected def encodeEdgeClass(edge: RawEdge): EdgeClass =
    EdgeClass.Edge // for now, we only have one edge class in the dataset

  def shuffleEdges(edges: List[RawEdge]): List[RawEdge]

  def linearizeEdges(shuffledEdges: List[RawEdge], shuffledNodes: List[RawNode]): (List[EdgeClass], List[Int], List[Int]) =
    val nodesPos = shuffledNodes.zipWithIndex.map:
      case (node, index) =>
        node.nodeName -> index
    .toMap
    val edgesClass = shuffledEdges.map(encodeEdgeClass)
    val edgesFrom = shuffledEdges.map(edge => nodesPos(edge.fromNode.nodeName))
    val edgesTo = shuffledEdges.map(edge => nodesPos(edge.toNode.nodeName))
    (edgesClass, edgesFrom, edgesTo)

  def toEdgeArray(edgesClass: List[EdgeClass], edgesFrom: List[Int], edgesTo: List[Int]): Array[Int] =
    val edgeArray = new Array[Int](edgesClass.length * 3 + 1)
    var i = 0
    while i < edgesClass.length do
      edgeArray(i * 3 + 0) = edgesClass(i).id
      // shift edge position by POS_OFFSET to make position tokens unique vocab items
      edgeArray(i * 3 + 1) = edgesFrom(i) + POS_OFFSET
      edgeArray(i * 3 + 2) = edgesTo(i) + POS_OFFSET
      i += 1
    edgeArray(edgesClass.length * 3) = EdgeClass.EndOfEdges.id
    edgeArray

  def toEdges(edgeTokens: List[Int], nodes: List[RawNode]): List[RawEdge] =
    edgeTokens.grouped(3).map: edge =>
      val edgeClass = EdgeClass.values.find(_.id == edge(0)) match
        case Some(value) => value
        case None        => throw new RuntimeException(f"Unknown edge class id: ${edge(0)}")
      val edgeFrom = nodes(edge(1) - POS_OFFSET)
      val edgeTo = nodes(edge(2) - POS_OFFSET)
      RawEdge(edgeFrom, edgeTo, edgeClass.name)
    .toList

object RandomNodeLinearization extends NodeLineraization:
  def shuffleNodes(nodes: List[RawNode]): List[RawNode] =
    scala.util.Random.shuffle(nodes)

object SortedNodeLinearization extends NodeLineraization:
  def shuffleNodes(nodes: List[RawNode]): List[RawNode] =
    // nodes.groupBy(_.nodeClass).mapValues(scala.util.Random.shuffle).values.flatten.toList
    nodes.sortBy(_.nodeClass.id).toList

object RandomEdgeLinearization extends EdgeLineraization:
  def shuffleEdges(edges: List[RawEdge]): List[RawEdge] =
    scala.util.Random.shuffle(edges)

object SortedEdgeLinearization extends EdgeLineraization:
  def shuffleEdges(edges: List[RawEdge]): List[RawEdge] =
    edges.sortBy(edge =>
      // first check fromNode than toNode
      (edge.fromNode.nodeClass.id, edge.toNode.nodeClass.id)
    )

object NoEdgeLinearization extends EdgeLineraization:
  def shuffleEdges(edges: List[RawEdge]): List[RawEdge] = List()

case class PaddedGraphLinearization(
    val nodeLineraization: NodeLineraization,
    val edgeLineraization: EdgeLineraization,
    val maxLength: Int = 128
) extends GraphLinearization:

  protected def toPaddedArray(nodeArray: Array[Int], edgeArray: Array[Int]): Array[Int] =
    val paddedArray = Array.fill(maxLength)(paddingValue)
    var j = 0
    for node <- nodeArray do
      paddedArray(j) = node
      j += 1
    for edge <- edgeArray do
      paddedArray(j) = edge
      j += 1
    paddedArray

  def linearize(graph: Graph): Tensor1[DecoderContext, Int] =
    val shuffledNodes = nodeLineraization.shuffleNodes(graph.nodes)
    val shuffledEdges = edgeLineraization.shuffleEdges(graph.edges)
    val (edgesClass, edgesFrom, edgesTo) = edgeLineraization.linearizeEdges(shuffledEdges, shuffledNodes)
    val nodeArray = nodeLineraization.toNodeArray(shuffledNodes)
    val edgeArray = edgeLineraization.toEdgeArray(edgesClass, edgesFrom, edgesTo)
    Tensor1(Axis[DecoderContext]).fromArray(
      toPaddedArray(nodeArray, edgeArray)
    )

  def strictUnlinearize(lineraizedGraph: Tensor1[DecoderContext, Int]): Try[Graph] =
    try
      val tokens = py"list(${lineraizedGraph.jaxValue})".as[List[Int]].takeWhile(_ != EdgeClass.EndOfEdges.id)
      val (nodeTokens, rest) = tokens.splitAt(tokens.indexOf(NodeClass.EndOfNodes.id))
      val edgeTokens = rest.tail // drop EndOfNodes token
      val nodes = nodeLineraization.toNodes(nodeTokens)
      val edges = edgeLineraization.toEdges(edgeTokens, nodes)
      Success(Graph(nodes, edges))
    catch
      case e: Throwable => Failure(e)

type RawPlan = Map[String, py.Dynamic]
case class RawEdge(fromNode: RawNode, toNode: RawNode, edgeClassName: String)
case class RawNode(nodeName: String, nodeClass: NodeClass)

case class Graph(nodes: List[RawNode], edges: List[RawEdge])

case class RawSample(imgPos: Int, graph: Graph)
case class Sample(image: Tensor[(Width, Height, Channel), Float], lineraizedGraph: Tensor1[DecoderContext, Int]):
  def toGPU =
    val deviceGPU = GPU.devices.head
    Sample(image.toDevice(deviceGPU), lineraizedGraph.toDevice(deviceGPU))

object ResPlanDataset:
  // 1. Load data via ScalaPy
  private val json = py.module("json")
  private val builtins = py.module("builtins")

  def loadRawPlans(dataPath: String = "res/plans/20/metadata.json"): List[RawSample] =
    val file = builtins.open(dataPath, "rb")
    val data =
      try
        json.load(file)
      catch
        case e: Throwable =>
          throw RuntimeException(s"Error loading data from $dataPath. Maybe forgot to run 'git lfs pull'? Original error: ${e.getMessage}")
      finally
        file.close()
    data.as[List[Map[String, py.Dynamic]]]
      .map: plan =>
        val dict = plan.as[Map[String, py.Dynamic]]
        val imgPos = dict("img_pos").as[Int]
        val pyNodes = dict("nodes")
        val pyEdges = dict("edges")
        val nodes = pyNodes.as[List[String]].map(name => RawNode(name, NodeClass.fromName(name)))
        val edges = pyEdges.as[List[py.Dynamic]].map: e =>
          RawEdge(
            nodes.find(node => node.nodeName == e.bracketAccess(0).as[String]).head,
            nodes.find(node => node.nodeName == e.bracketAccess(1).as[String]).head,
            e.bracketAccess(2).as[String]
          )
        RawSample(imgPos, Graph(nodes, edges))

case class ResPlanDataset(
    plans: List[RawSample],
    images: Tensor[(Data, Width, Height, Channel), Int],
    graphLinearization: GraphLinearization,
    normalizeImage: Tensor[(Width, Height, Channel), Int] => Tensor[(Width, Height, Channel), Float],
    sourceOfRandomness: Iterator[Random.Key],
    inifinite: Boolean = false
):
  private def rawSample2Sample(rawSample: RawSample, key: Random.Key): Sample =
    val linearizedGraph = graphLinearization.linearize(rawSample.graph)
    // val image = images.slice(Axis[Data].at(rawSample.imgPos)) // <-- leads to OOM
    val index = Tensor(Shape1(Axis[Data] -> 1)).fill(rawSample.imgPos)
    // slice with extra steps
    val image = images.take(Axis[Data])(index).flatten((Axis[Data], Axis[Width])).relabel(
      Axis[Data |*| Width] -> Axis[Width]
    )
    Sample(normalizeImage(image), linearizedGraph)

  private def planIterator =
    if inifinite
    then Iterator.continually(plans).flatten
    else plans.iterator

  def length: Int = plans.length

  def iterator: Iterator[Sample] =
    planIterator.zip(sourceOfRandomness).map(rawSample2Sample)
