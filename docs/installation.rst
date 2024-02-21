.. _installation:

============
Installation
============

Python Version
--------------

We recommend using the latest version of Python.


Virtual environments
--------------------

We recommend to use a a virtual environment to manage the project dependencies.

Create an environment
~~~~~~~~~~~~~~~~~~~~~

Create a project folder and a :file:`venv` folder within:

.. code-block:: sh

    mkdir myproject
    cd myproject
    python3 -m venv venv


Activate the environment
~~~~~~~~~~~~~~~~~~~~~~~~

Before you work on your project, activate the corresponding environment:

.. code-block:: sh

    . venv/bin/activate


Install dcms_engagement
-------------------------------------

Within the activated environment, use the following command to install
dcms_engagement:

.. code-block:: sh

    (venv) cd dcms_engagement
    (venv) pip install .
