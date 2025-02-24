import matplotlib.pyplot as plt
import numpy as np

from operators_POO import plot_analytical_result_sigma, import_frequency_sweep

def import_result(s):
    s = s+".txt"
    with open(s, "r") as f:
        frequency = list()
        results = list()
        for line in f:
            if "%" in line:
                # on saute la ligne
                continue
            data = line.split()
            frequency.append(data[0])
            results.append(data[1])
            frequency = [float(element) for element in frequency]
            results = [complex(element) for element in results]
    return frequency, results

geometry1 = "broken_cubic"
geometry2 = "small"
geometry = geometry1 + '_' + geometry2

s_ABC_comsol = 'COMSOL_data/' + geometry + '_COMSOL_results'
frequency, results_ABC_comsol = import_result(s_ABC_comsol)

s_PML_comsol = 'COMSOL_data/PML/PML_' + geometry
#_, results_PML_comsol = import_result(s_PML_comsol)

s_comsol_eq_sect = 'COMSOL_Eq/Eq_section/' + geometry
#_, results_PML_comsol_eq_sect = import_result(s_comsol_eq_sect)

s_comsol_eq_view = 'COMSOL_Eq/Eq_view/' + geometry
#_, results_PML_comsol_eq_view = import_result(s_comsol_eq_view)

ope  = 'b2p'
dimP = 3
s_noQ = 'no_Q_results/' + ope +'_' + geometry + '_' + str(dimP)
#_, results_noQ = import_result(s_noQ)

if geometry2 == 'small':
    lc = 8e-3
elif geometry2 == 'large':
    lc = 2e-2
s_classical = 'classical/classical_' + geometry + '_' + ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimP)
_, results_classical = import_frequency_sweep(s_classical)

fig, ax = plt.subplots(figsize=(16,9))
freqvec = np.arange(80, 2001, 20)     
plot_analytical_result_sigma(ax, freqvec, 0.1)

ax.plot(frequency, results_ABC_comsol, label = geometry + '_COMSOL')
#ax.plot(frequency, results_PML_comsol, label = 'PML_COMSOL')
#ax.plot(frequency, results_noQ, label = 'noQ')
#ax.plot(frequency, results_PML_comsol_eq_sect, label = 'COMSOL_eq_sect')
#ax.plot(frequency, results_PML_comsol_eq_view, label = 'COMSOL_eq_view')
ax.plot(frequency, results_classical, label = geometry + '_' + ope + '_' + str(lc) + '_' + str(dimP) + '_' + str(dimP))
ax.grid(True)
ax.set_title(geometry)
ax.legend(loc='upper left')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'$\sigma$')
plt.tight_layout()
plt.show()



