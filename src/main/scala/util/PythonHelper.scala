package resplan.util

import me.shadaj.scalapy.py

object PythonHelper:
  lazy val pyutil =
    PythonSetup.initialize
    py.module("pyutil")

  lazy val pyplot =
    PythonSetup.initialize
    py.module("matplotlib.pyplot")

/** Manages Python environment setup for DimWit.
  *
  * Handles extraction of Python helper modules from JAR resources and configuration of Python path for ScalaPy integration.
  */
object PythonSetup:

  private lazy val sys = py.module("sys")

  /** Initialize Python environment by extracting helper modules and configuring paths.
    *
    * This method:
    *   - Extracts jax_helper.py from JAR resources to a temporary directory
    *   - Adds the temp directory to Python's sys.path
    *   - Registers shutdown hook for cleanup
    *   - Falls back to development paths if running from source
    *
    * This is called lazily on first access to JAX modules and is safe to call multiple times (initialization happens only once).
    */
  lazy val initialize: Unit =
    // Extract jax_helper.py from JAR resources to a temporary directory
    val resourcePath = "/python/pyutil.py"
    val resourceStream = getClass.getResourceAsStream(resourcePath)

    assert(resourceStream != null)
    try
      val tempDir = java.nio.file.Files.createTempDirectory("dimwit-python")
      val targetFile = tempDir.resolve("pyutil.py")
      java.nio.file.Files.copy(
        resourceStream,
        targetFile,
        java.nio.file.StandardCopyOption.REPLACE_EXISTING
      )

      // Add the temp directory to Python path
      sys.path.append(tempDir.toAbsolutePath.toString)

      // Register shutdown hook to clean up temp directory
      Runtime.getRuntime.addShutdownHook(new Thread(() =>
        try
          java.nio.file.Files
            .walk(tempDir)
            .sorted(java.util.Comparator.reverseOrder())
            .forEach(java.nio.file.Files.delete)
        catch case _: Exception => () // Ignore cleanup errors
      ))
    finally resourceStream.close()
