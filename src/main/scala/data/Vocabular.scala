package resplan.data

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
