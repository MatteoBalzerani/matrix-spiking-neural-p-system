Matrix implementation of Spiking Neural P Systems for Image Classification
=================================================

This project takes the extended model of Spiking Neural P Systems (SN P systems) applied to image classification tasks used in https://github.com/SandroErba/spiking-p-system and contains:
- Matrix representation of the SNP Systems applied to image classification
- GPU implementation of the system
- Testing and timing methods for both matricial and classic SNP systems.
The framework supports both small example systems and a multilayer architecture designed for processing structured inputs such as images.

The project has been developed in collaboration between the University of Milano-Bicocca and the University of Verona.

The initial codebase is a fork of:
https://github.com/a1sabau/spiking-p-system

and is sequently an extract of me and @SandroErba's work in:
https://github.com/SandroErba/spiking-p-system

The framework includes:
- Extended SNP systems (multi-spike firing rules)
- White hole mechanism for the classical SNP system
- Inhibitory synapses with anti-spikes and its matricial implementation
- Lightweight architecture without synaptic weights

It can be used to run small SN P system examples or a multilayer model for image classification, with the structure illustrated below.

.. image:: SNPS_scheme.png
   :alt: Multilayer SN P system architecture
   :align: center
   :width: 600px

Installation
------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/MatteoBalzerani/matrix-spiking-neural-p-system.git
   cd matrix-spiking-neural-p-system

Create a virtual environment (optional but recommended):

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Linux / Mac
   venv\Scripts\activate     # Windows

Install dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Usage
-----

If a different entry point is used:

.. code-block:: bash

   python main.py

Experiments can be executed by varying parameters in the directly in the code.

