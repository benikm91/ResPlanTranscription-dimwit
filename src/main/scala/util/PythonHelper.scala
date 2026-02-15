package resplan.util

import me.shadaj.scalapy.py

object PythonHelper:

  lazy val pyplot =
    py.module("matplotlib.pyplot")

  lazy val np =
    py.module("numpy")

/** Manages Python environment setup for DimWit.
  *
  * Handles extraction of Python helper modules from JAR resources and configuration of Python path for ScalaPy integration.
  */
object PythonSetup:

  private lazy val sys = py.module("sys")
