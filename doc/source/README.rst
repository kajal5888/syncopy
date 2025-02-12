.. Syncopy documentation master file

.. title:: Syncopy Documentation

.. image:: _static/syncopy_logo.png
    :alt: Syncopy logo
    :height: 200px
    :align: center


Welcome to the Documentation of SyNCoPy!
========================================

SyNCoPy (**Sy**\stems **N**\euroscience **Co**\mputing in **Py**\thon, spelled Syncopy in the following)
is a Python toolkit for user-friendly, large-scale electrophysiology data analysis.
We strive to achieve the following goals:

1. Syncopy provides a full *open source* Python environment for reproducible
   electrophysiology data analysis.
2. Syncopy is *scalable* to accommodate *very large* datasets. It automatically
   makes use of available computing resources and is developed with built-in
   parallelism in mind.
3. Syncopy is *compatible* with the MATLAB toolbox `FieldTrip <http://www.fieldtriptoolbox.org/>`_.

Getting Started
---------------
- Prerequisites: :doc:`Install Syncopy </setup>`
- Jumping right in: :doc:`Quickstart Guide <quickstart/quickstart>`

Want to contribute or just curious how the sausage
is made? Take a look at our :doc:`Developer Guide <developer/developers>`.


In depth Guides and Tutorials
-----------------------------
* :doc:`Basic Concepts <user/concepts>`
* :doc:`Syncopy for FieldTrip Users <user/fieldtrip>`
* :doc:`Handling Data <user/data>`
* :doc:`Fooof <tutorials/fooof>`
* :doc:`Resampling <tutorials/resampling>`

  
Auto-generate API Docs
-----------------------
* :doc:`User API <user/user_api>`

Navigation
----------
* :doc:`Sitemap <sitemap>`
* :ref:`genindex`
* :ref:`search`

Contact
-------
To report bugs or ask questions please use our `GitHub issue tracker <https://github.com/esi-neuroscience/syncopy/issues>`_.
For general inquiries please contact syncopy (at) esi-frankfurt.de.

.. Any sections to be included in the Documentation dropdown menu have to be in the toctree

.. toctree::
   :hidden:

   quickstart/quickstart.rst
   setup
   user/concepts.rst
   user/fieldtrip.rst
   user/data.rst
   user/user_api.rst
   tutorials/fooof.rst
   tutorials/resampling.rst
   developer/developers.rst
