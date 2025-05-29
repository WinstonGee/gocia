import numpy as np
import random
import ast
from ase.db import connect
from gocia.interface import Interface
from gocia.ga import (
    get_eneFactor,
    get_matedFactor,
    get_matedFactor2,
    get_matedFactor3
)
from gocia.ga.crossover import crossover_snsSurf_2d_poly
from gocia.ensemble.comparator import srtDist_similar_zz
from ase.io import read, write


class PopulationCanonicalPoly:
    def __init__(
        self,
        substrate=None,
        gadb=None,
        popSize=20,
        zLim=None,
        chemNumDict=None,
        compParam=None,
        matingMethod=None,
        convergeCrit=None,
        simParam1=5e-4,
        simParam2=0.25
    ):

        if gadb is not None:
            self.gadb = connect(gadb)

        if type(substrate) is str:
            self.substrate = read(substrate)
        else:
            self.substrate = substrate

        if zLim is not None:
            self.zLim = zLim
        if zLim is None:
            self.zLim = Interface(self.substrate, self.substrate).zLim

        self.popSize = popSize

        if convergeCrit is None:
            self.convergeCrit = 5 * self.popSize
        else:
            self.convergeCrit = convergeCrit

        if chemNumDict is not None:
            self.chemNumDict = chemNumDict

        self.iniSize = len(self)

        self.simParam1 = simParam1
        self.simParam2 = simParam2

    def __len__(self):
        return len(self.gadb)

    def is_converged(self):
        gmid = self.get_GMrow().id
        if gmid < self.iniSize:
            return False
        if len(self) - gmid > self.convergeCrit:
            return True
        else:
            return False

    def is_converged2(self):
        gmid = eval(open('gmid').read())
        if gmid < self.iniSize:
            return False
        else:
            return len(self) - gmid > self.convergeCrit

    def initializeDB(self, sc=False):
        if sc:
            # Not tested for this SC version
            for i in range(len(self.gadb)):
                r = self.gadb.get(id=i+1)
                self.gadb.update(i+1, mated=0, alive=1, label='0 0 init', adsFrags=r.get('adsFrags'))
        else:
            for i in range(len(self.gadb)):
                r = self.gadb.get(id=i+1)
                self.gadb.update(i+1, mated=0, alive=1, label='0 0 init', adsFrags=r.get('adsFrags'))
        self.natural_selection()
        gm = self.get_GMrow()
        with open('gmid', 'w') as f:
            f.write(str(gm.id))

    def get_ID(self, condString):
        tmp = []
        for r in self.gadb.select(condString):
            tmp.append(r.id)
        return tmp

    def get_valueOf(self, valueString, idList):
        tmp = []
        for i in idList:
            tmp.append(
                self.gadb.get(id=i)[valueString]
            )
        return tmp

    def get_GMrow(self):
        eneList = self.get_valueOf('eV', self.get_ID('done=1'))
        self.Emin = min(eneList)
        row_min = None
        for r in self.gadb.select(eV=self.Emin):
            row_min = r
        return row_min

    def get_fitness(self, idList):
        tmpList = self.get_valueOf('eV', idList)
        f_ene = get_eneFactor(tmpList)
        tmpList = self.get_valueOf('mated', idList)
        f_mat = get_matedFactor3(tmpList)
        return f_ene * f_mat

    def choose_parents(self):
        aliveList = self.get_ID('alive=1')
        fitList = self.get_fitness(aliveList)
        weights = fitList / sum(fitList)
        parents = np.random.choice(aliveList, size=2, replace=False, p=weights)
        parents = [int(parents[0]), int(parents[1])]
        return parents

    def natural_selection(self):
        aliveList = self.get_ID('alive=1')
        if len(aliveList) > self.popSize:
            fitnessList = self.get_fitness(aliveList)
            aliveList = [x for _, x in sorted(zip(fitnessList, aliveList))]
            deadList = aliveList[:-self.popSize]
            for d in deadList:
                print('REST IN PEACE, %i!' % d)
                self.gadb.update(d, alive=0)

    def is_uniqueInPop(self, atoms, ene, eneCut=0.05):
        '''
        Check similarity against the current population
        '''
        aliveList = self.get_ID('alive=1')
        eneList = self.get_valueOf('eV', aliveList)
        isUnique = True
        for ii in range(len(aliveList)):
            a = self.gadb.get(id=aliveList[ii]).toatoms()
            if a.get_chemical_formula() == atoms.get_chemical_formula():
                if srtDist_similar_zz(atoms, a, self.simParam1, self.simParam2):
                    if eneCut > 0:
                        if -eneCut < eneList[ii] - ene < eneCut:
                            isUnique = False
                            break
                    else:
                        isUnique = False
                        break
        return isUnique

    def is_uniqueInAll(self, atoms, ene, eneCut=0.05):
        '''
        Check similarity against the all sampled structures
        '''
        aliveList = self.get_ID('done=1')
        eneList = self.get_valueOf('eV', aliveList)
        isUnique = True
        for ii in range(len(aliveList)):
            a = self.gadb.get(id=aliveList[ii]).toatoms()
            if a.get_chemical_formula() == atoms.get_chemical_formula():
                if srtDist_similar_zz(atoms, a, self.simParam1, self.simParam2):
                    if eneCut > 0:
                        if -eneCut < eneList[ii] - ene < eneCut:
                            isUnique = False
                            break
                    else:
                        isUnique = False
                        break
        return isUnique

    def is_uniqueInPop_geom(self, atoms):
        '''
        Check similarity against the all sampled structures
        '''
        aliveList = self.get_ID('alive=1')
        isUnique = True
        for ii in range(len(aliveList)):
            a = self.gadb.get(id=aliveList[ii]).toatoms()
            if a.get_chemical_formula() == atoms.get_chemical_formula():
                if srtDist_similar_zz(atoms, a, self.simParam1, self.simParam2):
                    isUnique = False
                    break
        return isUnique

    def is_uniqueInAll_geom(self, atoms):
        '''
        Check similarity against the all sampled structures
        '''
        aliveList = self.get_ID('done=1')
        isUnique = True
        for ii in range(len(aliveList)):
            a = self.gadb.get(id=aliveList[ii]).toatoms()
            if a.get_chemical_formula() == atoms.get_chemical_formula():
                if srtDist_similar_zz(atoms, a, self.simParam1, self.simParam2):
                    isUnique = False
                    break
        return isUnique

    def convertFragStrToInfo(self, fragStr):
        info = {}
        if fragStr:
            info['adsorbate_fragments'] = ast.literal_eval(fragStr)
        else:
            info['adsorbate_fragments'] = []
        return info

    def convertFragListToInfo(self, fragList):
        info = {}
        if fragList is not None:
            info['adsorbate_fragments'] = fragList
        else:
            info['adsorbate_fragments'] = []
        return info

    def gen_offspring(self, mutRate=0.3, rattleOn=True, moveOn=True, permuteOn=True, transOn=True, transVec=[[-2, 2], [-2, 2]]):
        kid, parent = None, None
        mater, pater = 0, 0
        while kid is None:
            mater, pater = self.choose_parents()
            matAtms = self.gadb.get(id=mater).toatoms()
            patAtms = self.gadb.get(id=pater).toatoms()
            matFragStr = self.gadb.get(id=mater).get('adsFrags')
            patFragStr = self.gadb.get(id=pater).get('adsFrags')
            matInfo = self.convertFragStrToInfo(matFragStr)
            patInfo = self.convertFragStrToInfo(patFragStr)
            surf1 = Interface(matAtms, self.substrate, zLim=self.zLim, info=matInfo) 
            surf2 = Interface(patAtms, self.substrate, zLim=self.zLim, info=patInfo)
            kid = crossover_snsSurf_2d_poly(surf1, surf2, tolerance=0.75)
            parent = surf1.copy()
        print('PARENTS: %i and %i' % (mater, pater))
        myMutate = ''
        # choices of mutation method
        mutList = [0,1,2,3]
        if srtDist_similar_zz(matAtms, patAtms) or not self.is_uniqueInAll_geom(kid.get_allAtoms()):
            print(' |- TOO SIMILAR!')
            mutRate = 1
            mutList = [0]
        if np.random.rand() < mutRate:
            mutType = np.random.choice(mutList, size=1)[0]
            if mutType == 0 and rattleOn:
                myMutate = 'rattle'
                kid.rattleMut_frag()
            if mutType == 1 and permuteOn:
                myMutate = 'permute'
                kid.permuteMut_frag()
            if mutType == 2 and transOn:
                myMutate = 'translate'
                kid.transMut(transVec=transVec)
            if mutType == 3 and moveOn:
                myMutate = 'move'
                kid.moveMut_frag([l for l in self.chemNumDict])
        open('label', 'w').write('%i %i %s' % (mater, pater, myMutate))
        self.gadb.update(mater, mated=self.gadb.get(id=mater).mated+1)
        self.gadb.update(pater, mated=self.gadb.get(id=pater).mated+1)
        open('fragments', 'w').write('%s' % kid.get_fragList() )
        return kid

    def gen_offspring_box(self, mutRate=0.3, xyzLims=[], bondRejList=None, constrainTop=False, rattleOn=True, moveOn=True, permuteOn=True, transOn=True, transVec=[[-2, 2], [-2, 2]]):
        kid, parent = None, None
        mater, pater = 0, 0
        while kid is None:
            mater, pater = self.choose_parents()
            matAtms = self.gadb.get(id=mater).toatoms()
            patAtms = self.gadb.get(id=pater).toatoms()
            matFragStr = self.gadb.get(id=mater).get('adsFrags')
            patFragStr = self.gadb.get(id=pater).get('adsFrags')
            matInfo = self.convertFragStrToInfo(matFragStr)
            patInfo = self.convertFragStrToInfo(patFragStr)
            surf1 = Interface(matAtms, self.substrate, zLim=self.zLim, info=matInfo) 
            surf2 = Interface(patAtms, self.substrate, zLim=self.zLim, info=patInfo)
            kid = crossover_snsSurf_2d_poly(surf1, surf2, tolerance=0.5, bondRejList=bondRejList, chemNumDict=self.chemNumDict)
            parent = random.choice([surf1, surf2]).copy()
            print('PARENTS: %i and %i' % (mater, pater))
        print(matInfo['adsorbate_fragments'], [matAtms[l].get_chemical_formula() for l in matInfo['adsorbate_fragments']])
        print(patInfo['adsorbate_fragments'], [patAtms[l].get_chemical_formula() for l in patInfo['adsorbate_fragments']])
        mutType = ''
        if srtDist_similar_zz(matAtms, patAtms)\
                or srtDist_similar_zz(matAtms, kid.get_allAtoms())\
                or srtDist_similar_zz(patAtms, kid.get_allAtoms()):
            print(' |- TOO SIMILAR!')
            mutRate = 1

        if np.random.rand() < mutRate:
            # collect the operations to use
            mutList = []
            if rattleOn: mutList.append('rattle')
            if moveOn: mutList.append('move')
            if permuteOn: mutList.append('permute')
            if transOn: mutList.append('translate')
            print('Mutation operator list: ', mutList)

            mutType = np.random.choice(mutList)
            if mutType == 'rattle':
                kid.rattleMut_frag()
                kid.rattleMut_buffer()
            if mutType == 'move':
                #kid.moveMut_frag([l for l in self.chemPotDict])
                myFrag = np.random.choice([l for l in self.chemNumDict], size=1)[0]
                kid.leachMut_frag([myFrag])
                # the grow step needs info of the constraints
                # otherwise very prone to dead loop!
                # kid.growMut_box_frag([myFrag], xyzLims=xyzLims,
                #                 bondRejList=bondRejList, constrainTop=constrainTop)
                tmpKid = None
                while tmpKid is None:
                    tmpKid = kid.copy()
                    # # growMut_box_frag() is messed up -- needs fixing
                    # tmpKid.growMut_box_frag([l for l in self.chemPotDict], xyzLims=xyzLims,
                    #             bondRejList=bondRejList, constrainTop=constrainTop)
                    tmpKid.growMut_frag([l for l in self.chemNumDict], bondRejList=bondRejList)
                kid = tmpKid.copy()
            if mutType == 'permute':
                kid.permuteMut_frag()
            if mutType == 'translate':
                kid.transMut(transVec=transVec)
        open('label', 'w').write('%i %i %s' % (mater, pater, mutType))
        self.gadb.update(mater, mated=self.gadb.get(id=mater).mated+1)
        self.gadb.update(pater, mated=self.gadb.get(id=pater).mated+1)
        open('fragments', 'w').write('%s' % kid.get_fragList() )
        return kid


    # TODO: connectivity checker and customizable "rules"
        # needed for cases where there are reactive events between frags during local opt
        # We'd expect bond between atoms in fragment and if not then broke bond
        # We'd also expect some bond between bridle atom and anchor atom, maybe more for bridle atom
            # ZZ: There may be changes in the binding site/mode. But if it is still on the usrface it should be fine
        # Could two fragments combine to make a new fragment? How evaluate?
            # ZZ: removing the substrate, and clustering into fragments by connectivity. Then decide.
        # How to calculate grand potential when there are new unexpected fragments???


    def add_vaspResult(self, vaspdir='.', isAlive=1):
        import os
        cwdFiles = os.listdir(vaspdir)
        if 'OSZICAR' in cwdFiles\
                and 'BADSTRUCTURE' not in cwdFiles\
                and 'ERROR' not in open('%s/OSZICAR' % vaspdir).read():
            if 'E0' in open(vaspdir+'/OSZICAR', 'r').readlines()[-1]:
                try:
                    s = read('%s/OUTCAR' % vaspdir, index='-1')
                except:
                    s = read('%s/vasprun.xml' % vaspdir, index='-1')
                dirname = os.getcwd().split('/')[-1]
                try:
                    mag = s.get_magnetic_moment()
                except:
                    oszicar_tail = open('OSZICAR').readlines()[-1]
                    if 'mag' in oszicar_tail:
                        mag = eval(oszicar_tail.split()[-1])
                    else:
                        mag = 0
                ene_eV = s.get_potential_energy()

                # read or detect the fragment list
                try:
                    myFragList = eval(open('%s/fragments' % vaspdir, 'r').readlines()[0].rstrip('\n'))
                    if max([i for f in myFragList for i in f]) >= len(s):
                        print('Bad fragList! Detecting from connectivity...')
                        myFragList = None
                except:
                    print('No fragList! Detecting from connectivity...')
                    myFragList = None
                if myFragList == None:
                    surf_tmp = Interface(s, self.substrate)
                    myFragList = surf_tmp.detect_fragList()
                    del surf_tmp
                myInfo = self.convertFragListToInfo(myFragList)
                myFragList = str(myFragList)

                myLabel = open('label', 'r').read()
                print('\n%s IS BORN with E = %.3f eV\t[%s]' % (
                    dirname, ene_eV, myLabel))
                if self.is_uniqueInAll(s, ene_eV):
                    if ene_eV < self.get_GMrow()['eV']:
                        print(f' |- {dirname} is the new GM!')
                        with open('gmid', 'w') as f:
                            f.write(str(len(self)))
                    self.gadb.write(
                        s,
                        name=dirname,
                        mag=mag,
                        eV=ene_eV,
                        mated=0,
                        done=1,
                        alive=isAlive,
                        label=myLabel,
                        adsFrags=myFragList 
                    )
                else:
                    print(f' |- {dirname} is a duplicate!')
                    self.gadb.write(
                        s,
                        name=dirname,
                        mag=mag,
                        eV=ene_eV,
                        mated=0,
                        done=1,
                        alive=0,
                        label=myLabel,
                        adsFrags=myFragList 
                    )

    def add_aseResult(self, atoms, workdir='.', fn_frag=None, isAlive=1):
        # read or detect the fragment list
        try:
            myFragList = eval(open(fn_frag, 'r').readlines()[0].rstrip('\n'))
            if max([i for f in myFragList for i in f]) >= len(atoms):
                print('Bad fragList! Detecting from connectivity...')
                myFragList = None
        except:
            print('No fragList! Detecting from connectivity...')
            myFragList = None
        if myFragList == None:
            surf_tmp = Interface(atoms, self.substrate)
            myFragList = surf_tmp.detect_fragList()
            del surf_tmp
        atoms.info['adsFrag']=myFragList
        myInfo = self.convertFragListToInfo(myFragList)
        myFragList = str(myFragList)
        

        ene_eV = atoms.get_potential_energy()
        try:
            mag = atoms.get_magnetic_moments().sum()
        except:
            mag = 0

        myLabel = open(f'label', 'r').read()
        print('\n%s IS BORN with E = %.3f eV\t[%s]' % (
            workdir, ene_eV, myLabel))
        if self.is_uniqueInAll(atoms, ene_eV):
            if ene_eV < self.get_GMrow()['eV'] and isAlive == 1:
                print(f' |- {workdir} is the new GM!')
                with open('../gmid', 'w') as f:
                    f.write(str(len(self)))
            self.gadb.write(
                atoms,
                name=workdir,
                mag=mag,
                eV=ene_eV,
                mated=0,
                done=1,
                alive=isAlive,
                label=myLabel,
                adsFrags=myFragList 
            )
        else:
            print(f' |- {workdir} is a duplicate!')
            self.gadb.write(
                atoms,
                name=workdir,
                mag=mag,
                eV=ene_eV,
                mated=0,
                done=1,
                alive=0,
                label=myLabel,
                adsFrags=myFragList 
            )