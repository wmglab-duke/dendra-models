from dendra.models.mechanisms._mechanism import Mechanism as M


class leak(M):
    M.GLOBAL(gkleak=0.0, gnaleak=0.0, gcaleak=0.0)

    M.USEION("na", read=["ena"], write=["ina"])
    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("ca", read=["eca"], write=["ica"])

    def ina(self, v):
        return self.gnaleak * (v - self.ena)

    def ik(self, v):
        return self.gkleak * (v - self.ek)

    def ica(self, v):
        return self.gcaleak * (v - self.eca)