from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
from fairchem.core import OCPCalculator
import numpy as np
import math
import os
import phonopy
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure
import torch
import glob
from multiprocessing import Pool


class MLFFWorker():
    def __init__(self, material_path=None):
        # self.oclimax = OCLIMAX()
        self.material_path = material_path
        self.material_name = os.path.basename(material_path)
        self.struc_path = os.path.join(self.material_path, 'POSCAR-unitcell')
        self.struc = read(self.struc_path, format="vasp")
        self.mlff_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mlff')
        self.dft_yaml_path = os.path.join(self.material_path, 'phonopy.yaml')
        self.dft_forcesets_path = os.path.join(self.material_path, 'FORCE_SETS')
        self.nx = self.ny = self.nz = None

    def run_opt_and_dos(self, struc=None, potential_index=0,lmin=12.0,fmax=0.01,nmax=100,delta=0.03):
        struc = self.struc if struc is None else struc
        try:
            lmin = float(lmin)
        except:
            lmin = 12.0
        try:
            fmax = float(fmax)
        except:
            fmax = 0.001
        try:
            nmax = int(nmax)
        except:
            nmax = 100
        try:
            delta = float(delta)
        except:
            delta = 0.03
        abc = struc.cell.cellpar()[0:3]
        nx = math.ceil(lmin/abc[0])
        ny = math.ceil(lmin/abc[1])
        nz = math.ceil(lmin/abc[2])
        self.nx = nx
        self.ny = ny
        self.nz = nz
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        use_cpu = device == 'cpu'
        torch.set_default_dtype(torch.float32)
        print('INFO: Running structural optimization...')
        if potential_index == 6:    # Fairchem eqV2 S DeNS
            calculator = OCPCalculator(
            checkpoint_path="eqV2_153M_omat_mp_salex.pt",
            cpu=use_cpu,
            )
            struc.set_calculator(calculator)
            dyn = FIRE(struc)
            dyn.run(fmax=fmax, steps=nmax)
            atoms_relaxed = dyn.atoms.copy()

        elif potential_index == 12:   # eSEN-30M-OAM
            calculator = OCPCalculator(checkpoint_path="esen_30m_oam.pt", cpu=use_cpu)
            struc.set_calculator(calculator)
            dyn = FIRE(struc)
            dyn.run(fmax=fmax, steps=nmax)
            atoms_relaxed = dyn.atoms.copy()

        write(f"{self.material_name}_{potential_index}_POSCAR-unitcell", atoms_relaxed, direct=True, format='vasp')
        print('INFO: Structural optimization finished.')

        # Phonon calculator
        npc = len(struc.numbers)
        nsc = npc*nx*ny*nz
        print('INFO: Number of atoms in unit cell: '+str(npc))
        print('INFO: Supercell dimension: '+' '.join(map(str,[nx,ny,nz])))
        print('INFO: Total number of atoms in supercell: '+str(nsc))

        unitcell, _ = read_crystal_structure(f"{self.material_name}_{potential_index}_POSCAR-unitcell", interface_mode='vasp')
        phonon = Phonopy(unitcell,
                         supercell_matrix=[self.nx, self.ny, self.nz],
                         primitive_matrix=np.array([[1,0,0], [0,1,0], [0,0,1]]))
        os.remove(f"{self.material_name}_{potential_index}_POSCAR-unitcell")

        # Phonon calculation with phonopy

        phonon.generate_displacements(distance=delta)
        supercells = phonon.supercells_with_displacements
        ns = len(supercells)
        print('INFO: Total number of displacements: '+str(ns))

        sets_of_forces=[]
        for i in range(ns):
            supercell = supercells[i]
            sc = Atoms(symbols=supercell.get_chemical_symbols(),
                       scaled_positions=supercell.get_scaled_positions(),
                       cell=supercell.get_cell(),
                       pbc=True)
            calculator.calculate(atoms=sc,properties='forces', system_changes=None)
            sets_of_forces.append(calculator.results['forces'])
            if i<ns-1:
                print('INFO: '+str(i+1)+' of '+str(ns)+' displacements finished', end ='\r')
            else:
                print('INFO: '+str(i+1)+' of '+str(ns)+' displacements finished')

        phonon.forces = sets_of_forces
        phonon.produce_force_constants()

        saved_fc_name = self.material_path + f"/phonopy_info_{potential_index}.yaml"
        phonon.save(saved_fc_name, settings={'force_constants': True})
        
    def run_dft_dos(self):
        phonon = phonopy.load(self.dft_yaml_path, is_nac=False, force_sets_filename=self.dft_forcesets_path)
        phonon.produce_force_constants()
        saved_fc_name = self.material_path + f"/phonopy_info_dft.yaml"
        phonon.save(saved_fc_name, settings={'force_constants': True})


def gen_phonon_mesh_dict(material_path):
    mlff = MLFFWorker(material_path=material_path)
    print(mlff.material_name)
    for i in [6, 12]:
        mlff.run_opt_and_dos(potential_index = i)

if __name__ == "__main__":

    material_list = glob.glob("/home/xb2/10Tdisk/projects/ml_phonon/crystals/test_phonondb/phonon_benchmark_database/*")

    parallel = False

    if parallel:
    # Create a pool of workers, the size of the pool is by default the number of CPU cores
        with Pool(processes=8) as pool:
            # Map the function to the inputs and collect the results
            # pool.map is suitable for functions with a single argument
            results = pool.map(gen_phonon_mesh_dict, material_list)
            
        # Optional: Do something with the results
        # ...
        
    else:
        for i in material_list:
            gen_phonon_mesh_dict(i)

