First the user should have the *S3SAX01_sqlite3.db* file (from SKADS-Sky-Simulation) downloaded in their computer, and the db_path in query_data_lib.py should be altered to its corresponding path.

The main file is analysis.py, which can be edited to perform different analysis with different parameters: *python analysis.py*

Notes:
1. The continuum still has to be added to the simulated spectra
2. The galaxies queried from the database with null or equal central and peak luminosity densities still have to be correctly treated (for now they are being ignored, but they still enter into the spectra counts)