import scipy.io
from pathlib import Path

from gen_simIMG2 import gen_simIMG2


data = scipy.io.loadmat('Breast_Data_partial.mat')

for n in range(1, 22):
    simImg, smap, parMap, temp, ID, aif, T10, aifci_1s, cts = gen_simIMG2(
        data['new_data'], n
    )

    output_dir = Path(ID)
    if not output_dir.exists():
        output_dir.mkdir()
    else:
        print(f"{ID} already exists.")

    scipy.io.savemat(output_dir / 'cart_images.mat', {'simImg': simImg})
    scipy.io.savemat(output_dir / 'coil_sens.mat', {'smap': smap})
