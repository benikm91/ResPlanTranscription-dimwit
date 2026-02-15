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
    javaOptions ++= Seq(
      // "-XX:G1PeriodicGCInterval=1000"
      "-XX:+UseZGC",
      "-XX:ZCollectionInterval=1" // Forces a GC cycle every 1 second, regardless of heap usage
    )
  )

// Ensure local ivy resolver is included
fork := true
resolvers += Resolver.defaultLocal
envVars ++= sys.env
