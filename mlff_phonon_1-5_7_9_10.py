from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
import matgl
from matgl.ext.ase import PESCalculator, Relaxer
from pymatgen.io.ase import AseAtomsAdaptor
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from sevenn.sevennet_calculator import SevenNetCalculator
import numpy as np
import math
import os
import phonopy
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure
import torch
import glob

from pymatgen.io.ase import AseAtomsAdaptor
from multiprocessing import Pool

from mattersim.forcefield.potential import MatterSimCalculator
from tensorpotential.calculator import grace_fm

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
        self.mesh_path = os.path.join(self.material_path, 'mesh.conf')

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
        torch.set_default_dtype(torch.float32)
        print('INFO: Running structural optimization...')
        if potential_index == 1:   # CHGNet
            pot = matgl.load_model("CHGNet-MPtrj-2024.2.13-PES-11M")
            calculator = PESCalculator(pot)
            relaxer = Relaxer(potential=pot,relax_cell=False)
            relax_results = relaxer.relax(struc, fmax=fmax, steps=nmax, verbose=True)
            final_structure = relax_results['final_structure']
            atoms_relaxed = AseAtomsAdaptor().get_atoms(final_structure)
        elif potential_index == 2:   # M3GNet
            pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            calculator = PESCalculator(pot)
            relaxer = Relaxer(potential=pot,relax_cell=False)
            relax_results = relaxer.relax(struc, fmax=fmax, steps=nmax, verbose=True)
            final_structure = relax_results['final_structure']
            atoms_relaxed = AseAtomsAdaptor().get_atoms(final_structure)

        elif potential_index == 3:   # ORBNN
            orbff = pretrained.orb_v1(weights_path="orbff-v1-20240827.ckpt", device=device)
            calculator = ORBCalculator(orbff, device=device)
            struc.set_calculator(calculator)
            dyn = FIRE(struc)
            dyn.run(fmax=fmax, steps=nmax)
            atoms_relaxed = dyn.atoms.copy()

        elif potential_index == 4:   # SeveNN
            calculator = SevenNetCalculator("SevenNet-0_11July2024", device=device)
            struc.set_calculator(calculator)
            dyn = FIRE(struc)
            dyn.run(fmax=fmax, steps=nmax)
            atoms_relaxed = dyn.atoms.copy()

        elif potential_index == 5:   # MatterSim 5M
            calculator = MatterSimCalculator(load_path="mattersim-v1.0.0-5M.pth")
            struc.set_calculator(calculator)
            dyn = FIRE(struc)
            dyn.run(fmax=fmax, steps=nmax)
            atoms_relaxed = dyn.atoms.copy()

        elif potential_index == 7:   # MatterSim 1M
            calculator = MatterSimCalculator(load_path="mattersim-v1.0.0-1M.pth")
            struc.set_calculator(calculator)
            dyn = FIRE(struc)
            dyn.run(fmax=fmax, steps=nmax)
            atoms_relaxed = dyn.atoms.copy()

        elif potential_index == 9:   # SeveNN
            calculator = SevenNetCalculator("7net-mf-ompa", modal='mpa', device=device)
            struc.set_calculator(calculator)
            dyn = FIRE(struc)
            dyn.run(fmax=fmax, steps=nmax)
            atoms_relaxed = dyn.atoms.copy()

        elif potential_index == 10:   # GRACE
            calculator = grace_fm('GRACE-2L-OAM')
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
            calculator.calculate(atoms=sc,properties='forces')
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
        super_cell_matrix = np.loadtxt(self.mesh_path, dtype=int, max_rows=1, usecols=(2,3,4))
        phonon = phonopy.load(unitcell_filename=self.struc_path, supercell_matrix=super_cell_matrix, is_nac=False, force_sets_filename=self.dft_forcesets_path)
        phonon.produce_force_constants()
        saved_fc_name = self.material_path + f"/phonopy_info_dft.yaml"
        phonon.save(saved_fc_name, settings={'force_constants': True})


def gen_phonon_mesh_dict(material_path):
    mlff = MLFFWorker(material_path=material_path)
    print(mlff.material_name)
    for i in [1, 2, 3, 4, 5, 7, 9, 10]:
        mlff.run_opt_and_dos(potential_index = i)
    mlff.run_dft_dos()

if __name__ == "__main__":

    material_list = glob.glob("/home/xb2/10Tdisk/projects/ml_phonon/crystals/test_phonondb/phonon_benchmark_database/*")

    parallel = False

    if parallel:
    # Create a pool of workers, the size of the pool is by default the number of CPU cores
        with Pool(processes=8) as pool:
            # Map the function to the inputs and collect the results
            # pool.map is suitable for functions with a single argument
            results = pool.map(gen_phonon_mesh_dict, material_list)
        
    else:
        for i in material_list:
            gen_phonon_mesh_dict(i)


