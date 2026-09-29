"""Who published each entry of the layer-2 registries, as bibliography keys.

Plain literals only: ``docs/tools/sync_bib.py`` reads this file with ``ast`` and
must not import JAX to learn which references the documentation needs.

Every key is an entry of the technical paper's ``references.bib``, copied into
``docs/references.bib`` by ``sync_bib.py``.  ``ggah_tables`` raises if a registry
gains an entry that is missing here, so a fit cannot reach the documentation
without its source.
"""

#: ``ggah_mod.halos.mass_function.MULTIPLICITY``.  The two ``_csst`` entries are
#: Tinker et al. (2008) with its coefficients rescaled per cosmology by the
#: ``emu_hmf`` network, trained on the CSST emulator of Chen & Yu (2025).
MULTIPLICITY = {
    "press74": ("PressSchechter_1974ApJ...187..425P",),
    "sheth99": ("ShethTormen_1999MNRAS.308..119S",),
    "jenkins01": ("JenkinsFrenkWhite_2001MNRAS.321..372J",),
    "warren06": ("WarrenAbazajianHolz_2006ApJ...646..881W",),
    "tinker08": ("TinkerKravtsovKlypin_2008ApJ...688..709T",),
    "crocce10": ("CrocceFosalbaCastander_2010MNRAS.403.1353C",),
    "bhattacharya11": ("BhattacharyaHeitmannWhite_2011ApJ...732..122B",),
    "watson13": ("WatsonIlievDAloisio_2013MNRAS.433.1230W",),
    "angulo12": ("AnguloSpringelWhite_2012MNRAS.426.2046A",),
    "bocquet16": ("BocquetSaroDolag_2016MNRAS.456.2361B",),
    "despali16": ("DespaliGiocoliAngulo_2016MNRAS.456.2486D",),
    "rodriguezpuebla16": ("Rodriguez-PueblaBehrooziPrimack_2016MNRAS.462..893R",),
    "comparat17": ("ComparatPradaYepes_2017MNRAS.469.4157C",),
    "seppi20": ("SeppiComparatNandra_2021A&A...652A.155S",),
    "yung24": ("YungSomervilleNguyen_2024MNRAS.530.4868Y",),
    "yung25": ("YungSomervilleIyer_2025MNRAS.543.3802Y",),
    "tinker08_csst": ("TinkerKravtsovKlypin_2008ApJ...688..709T",
                      "ChenYu_2025SCPMA..6809513C"),
    "tinker08_csst_vir": ("TinkerKravtsovKlypin_2008ApJ...688..709T",
                          "ChenYu_2025SCPMA..6809513C"),
}

#: ``ggah_mod.halos.linear_bias.BIAS``.  ``press74`` is the peak-background-split
#: bias of the Press & Schechter (1974) mass function, derived by Mo & White (1996).
BIAS = {
    "tinker10": ("TinkerRobertsonKravtsov_2010ApJ...724..878T",),
    "sheth99": ("ShethTormen_1999MNRAS.308..119S",),
    "press74": ("PressSchechter_1974ApJ...187..425P",
                "MoWhite_1996MNRAS.282..347M"),
    "bhattacharya11": ("BhattacharyaHeitmannWhite_2011ApJ...732..122B",),
    "sheth01": ("ShethMoTormen_2001MNRAS.323....1S",),
    "comparat17": ("ComparatPradaYepes_2017MNRAS.469.4157C",),
}

#: ``ggah_mod.halos.concentration.CONCENTRATION``.
CONCENTRATION = {
    "duffy08": ("DuffySchayeKay_2008MNRAS.390L..64D",),
    "dutton14": ("DuttonMaccio_2014MNRAS.441.3359D",),
    "klypin16": ("KlypinYepesGottlober_2016MNRAS.457.4340K",),
    "bhattacharya13": ("BhattacharyaHabibHeitmann_2013ApJ...766...32B",),
    "diemer19": ("DiemerJoyce_2019ApJ...871..168D",),
    "seppi21": ("SeppiComparatNandra_2021A&A...652A.155S",),
}
