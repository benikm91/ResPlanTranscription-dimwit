package resplan

import dimwit.*
import resplan.data.ResPlanDataset
import resplan.data.RandomGraphLinearization
import resplan.data.StaticRenderer
import resplan.util.PythonHelper

@main def hello(): Unit =
  val dataset = ResPlanDataset(
    ResPlanDataset.loadRawPlans("res/ResPlan.pkl"),
    RandomGraphLinearization,
    StaticRenderer
  )
  println(dataset.length)
  val sample = dataset.iterator.next
  val plt = PythonHelper.pyplot
  plt.imshow(sample.image.jaxValue)
  plt.show()
