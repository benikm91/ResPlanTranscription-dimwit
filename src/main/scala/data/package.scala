package resplan

import dimwit.Label

package object data:

  trait Data derives Label
  trait Width derives Label
  trait Height derives Label
  trait Channel derives Label
  trait Node derives Label
  trait Edge derives Label

  val paddingValue = -1
