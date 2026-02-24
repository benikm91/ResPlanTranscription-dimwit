package resplan.nn

trait ParamsFor[Module]:

  def toModule: Module
