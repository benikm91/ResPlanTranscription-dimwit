package resplan.util

import dimwit.random.Random

object RandomUtil:
  extension (key: Random.Key)
    def toSourceOfRandomness: Iterator[Random.Key] =
      Iterator.iterate(key.split2())((_, restKey) => restKey.split2()).map((key, _) => key)
