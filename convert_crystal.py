import sys
import argparse
import os

# Common elements and their approximate atomic masses
MASS_TO_ELEMENT = {
    1.008: 'H', 4.003: 'He', 12.011: 'C', 14.007: 'N', 15.999: 'O',
    24.305: 'Mg', 26.982: 'Al', 28.085: 'Si', 39.098: 'K', 40.078: 'Ca',
    55.845: 'Fe', 58.693: 'Ni', 63.546: 'Cu', 183.84: 'W'
}

ELEMENT_TO_MASS = {v: k for k, v in MASS_TO_ELEMENT.items()}

def guess_element_from_mass(mass):
    best_element = None
    min_diff = float('inf')
    for m, el in MASS_TO_ELEMENT.items():
        diff = abs(m - mass)
        if diff < min_diff and diff < 1.0: # arbitrary threshold
            min_diff = diff
            best_element = el
    if best_element:
        return best_element
    return "Na"

def crystal_to_data(in_file, out_file):
    atoms = []
    elements_found = []
    cell = [0.0, 0.0, 0.0]
    
    with open(in_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
            if line.startswith('# cell_1:'):
                parts = line.split()
                cell[0] = float(parts[2])
            elif line.startswith('# cell_2:'):
                parts = line.split()
                cell[1] = float(parts[3])
            elif line.startswith('# cell_3:'):
                parts = line.split()
                cell[2] = float(parts[4])
        else:
            parts = line.split()
            if len(parts) >= 4:
                element = parts[0]
                x, y, z = map(float, parts[1:4])
                if element not in elements_found:
                    elements_found.append(element)
                type_id = elements_found.index(element) + 1
                atoms.append((type_id, x, y, z))
                
    num_atoms = len(atoms)
    num_types = len(elements_found)
    
    with open(out_file, 'w') as f:
        f.write(f"LAMMPS data file generated from {os.path.basename(in_file)}\n\n")
        f.write(f"{num_atoms} atoms\n")
        f.write(f"{num_types} atom types\n\n")
        
        f.write(f"0.00000000 {cell[0]:.8f} xlo xhi\n")
        f.write(f"0.00000000 {cell[1]:.8f} ylo yhi\n")
        f.write(f"0.00000000 {cell[2]:.8f} zlo zhi\n\n")
        
        f.write("Masses\n\n")
        for i, el in enumerate(elements_found):
            mass = ELEMENT_TO_MASS.get(el, 1.0)
            f.write(f"  {i+1} {mass:.6f}\n")
            
        f.write("\nAtoms\n\n")
        for i, atom in enumerate(atoms):
            f.write(f"{i+1} {atom[0]} {atom[1]:.8f} {atom[2]:.8f} {atom[3]:.8f}\n")

    print(f"Converted {in_file} to {out_file} ({num_atoms} atoms, {num_types} types).")

def data_to_crystal(in_file, out_file, crystal_system="cubic"):
    atoms = []
    type_to_element = {}
    box = [0.0, 0.0, 0.0]
    
    with open(in_file, 'r') as f:
        lines = f.readlines()
        
    num_atoms = 0
    section = None
    
    KEYWORDS = [
        "Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
        "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
        "Improper Coeffs", "BondBond Coeffs", "BondAngle Coeffs", "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs", "AngleTorsion Coeffs", "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs", "AngleAngle Coeffs"
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        
        if 'atoms' in stripped and len(stripped.split()) == 2:
            num_atoms = int(stripped.split()[0])
            continue
        if 'atom types' in stripped:
            continue
            
        if 'xlo xhi' in stripped:
            parts = stripped.split()
            box[0] = float(parts[1]) - float(parts[0])
            continue
        if 'ylo yhi' in stripped:
            parts = stripped.split()
            box[1] = float(parts[1]) - float(parts[0])
            continue
        if 'zlo zhi' in stripped:
            parts = stripped.split()
            box[2] = float(parts[1]) - float(parts[0])
            continue
            
        # Check if the line starts with any of the recognized LAMMPS section keywords
        is_keyword = False
        for kw in KEYWORDS:
            if stripped.startswith(kw):
                section = kw.split()[0] # Switch to this section (e.g., "Masses", "Atoms")
                is_keyword = True
                break
                
        if is_keyword:
            continue
            
        if section == "Masses":
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    type_id = int(parts[0])
                    mass = float(parts[1])
                    type_to_element[type_id] = guess_element_from_mass(mass)
                except ValueError:
                    pass
                
        elif section == "Atoms":
            parts = stripped.split()
            if len(parts) >= 5:
                # Format typically: id type x y z [nx ny nz]
                # Assuming 'atomic' style which is standard for simple crystals.
                # Type is index 1, x y z are indices 2 3 4.
                try:
                    type_id = int(parts[1])
                    x, y, z = map(float, parts[2:5])
                    atoms.append((type_id, x, y, z))
                except ValueError:
                    pass
                    
    element_counts = {}
    for atom in atoms:
        # Default to the type ID string if the type wasn't found in a Masses section
        el = type_to_element.get(atom[0], str(atom[0]))
        element_counts[el] = element_counts.get(el, 0) + 1
        
    formula = "".join([f"{k}{v}" for k, v in element_counts.items()])
    
    with open(out_file, 'w') as f:
        f.write(f"# POLY crystal: converted  |  {len(atoms)} atoms  |  formula={formula}\n")
        f.write(f"# crystal_system: {crystal_system}\n")
        f.write(f"# cell_1: {box[0]:.8f} 0.00000000 0.00000000\n")
        f.write(f"# cell_2: 0.00000000 {box[1]:.8f} 0.00000000\n")
        f.write(f"# cell_3: 0.00000000 0.00000000 {box[2]:.8f}\n")
        f.write(f"# element  x  y  z\n")
        
        for atom in atoms:
            el = type_to_element.get(atom[0], str(atom[0]))
            f.write(f" {el}  {atom[1]:.8f}  {atom[2]:.8f}  {atom[3]:.8f}\n")
            
    print(f"Converted {in_file} to {out_file} ({len(atoms)} atoms).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between .crystal and LAMMPS .data files")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("-o", "--output", help="Output file path (optional, auto-generated if omitted)")
    parser.add_argument("--crystal_system", default="cubic", help="Crystal system to use when converting to .crystal (default: cubic)")
    
    args = parser.parse_args()
    
    in_file = args.input
    out_file = args.output
    
    if in_file.endswith(".crystal"):
        if not out_file:
            out_file = in_file.replace(".crystal", ".data")
        crystal_to_data(in_file, out_file)
    elif in_file.endswith(".data"):
        if not out_file:
            out_file = in_file.replace(".data", ".crystal")
        data_to_crystal(in_file, out_file, args.crystal_system)
    else:
        print("Error: Input file must end with .crystal or .data")
        sys.exit(1)
