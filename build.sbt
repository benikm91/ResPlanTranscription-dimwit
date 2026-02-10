val scala3Version = "3.8.1"

import scala.concurrent.duration.DurationInt
import lmcoursier.definitions.CachePolicy

csrConfiguration := csrConfiguration.value
  .withTtl(Some(0.seconds))
  .withCachePolicies(Vector(CachePolicy.LocalOnly))

lazy val root = project
  .in(file("."))
  .settings(
    name := "resplan",
    version := "0.1.0-SNAPSHOT",
    scalaVersion := scala3Version,
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % "1.0.0" % Test,
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "ch.contrafactus" %% "dimwit-core" % "0.1.0-SNAPSHOT" changing (),
      "ch.contrafactus" %% "dimwit-nn" % "0.1.0-SNAPSHOT" changing ()
    ),
    Compile / resourceDirectory := baseDirectory.value / "src" / "main" / "resources",
    javaOptions ++= {
      if (sys.props("os.name").toLowerCase.contains("mac")) {
        Seq("-XstartOnFirstThread") // For MacOS to run Python with GUI support
      } else {
        Seq.empty
      }
    }
  )

fork := true

// Ensure local ivy resolver is included
resolvers += Resolver.defaultLocal
