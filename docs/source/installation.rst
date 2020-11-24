Installation
============


This is the recommended way to install Pyedra.

Installing  with pip
^^^^^^^^^^^^^^^^^^^^

Make sure that the Python interpreter can load Pyedra code.
The most convenient way to do this is to use virtualenv, virtualenvwrapper, and pip.

After setting up and activating the virtualenv, run the following command:

.. code-block:: console

   $ pip install Pyedra
   ...

That should be it all.



Installing the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you’d like to be able to update your Pyedra code occasionally with the
latest bug fixes and improvements, follow these instructions:

Make sure that you have Git installed and that you can run its commands from a shell.
(Enter *git help* at a shell prompt to test this.)

Check out Pyedra main development branch like so:

.. code-block:: console

   $ git clone https://github.com/milicolazo/Pyedra.git
   ...

This will create a directory *Pyedra* in your current directory.

Then you can proceed to install with the commands

.. code-block:: console

   $ cd Pyedra
   $ pip install -e .
   ...

