# import copernicusmarine

# copernicusmarine.subset(
#   dataset_id="cmems_mod_ibi_wav_my_0.027deg_PT1H-i",
#   dataset_version="202411",
#   variables=["VHM0_WW", "VCMX", "VHM0", "VHM0_SW1", "VHM0_SW2", "VMDR", "VMDR_SW1", "VMDR_SW2", "VMDR_WW", "VMXL", "VPED", "VSDX", "VSDY", "VTM01_SW1", "VTM01_SW2", "VTM01_WW", "VTM02", "VTM10", "VTPK"],
#   minimum_longitude=-11,
#   maximum_longitude=-8.5,
#   minimum_latitude=38.5,
#   maximum_latitude=40.5,
#   start_datetime="2020-01-01T00:00:00",
#   end_datetime="2023-12-30T23:00:00",
#   coordinates_selection_method="strict-inside",
#   netcdf_compression_level=1,
#   disable_progress_bar=False,

# )

# import copernicusmarine

# copernicusmarine.subset(
#   dataset_id="cmems_mod_ibi_wav_my-aflux_0.027deg_P1H-i",
#   dataset_version="202411",
#   variables=["TLA", "TAUX", "TAUY"],
#   minimum_longitude=-11,
#   maximum_longitude=-8.5,
#   minimum_latitude=38.5,
#   maximum_latitude=40.5,
#   start_datetime="2020-01-01T00:00:00",
#   end_datetime="2023-12-30T23:00:00",
#   coordinates_selection_method="strict-inside",
#   netcdf_compression_level=1,
#   disable_progress_bar=True,
# )

# TAUX     -Surface downward eastward stress due to ocean viscous dissipation
# TAUY     -Surface downward northward stress due to ocean viscous dissipation
# TLA      -Wave mixing energy flux into sea water