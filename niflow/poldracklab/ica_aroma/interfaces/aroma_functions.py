import numpy as np
from past.utils import old_div
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, Directory, isdefined,
    SimpleInterface
)


class FeatureTimeSeriesInputSpec(BaseInterfaceInputSpec):
    melmix = File(exists=True, desc="melodic_mix text file")
    mc = File(exists=True, desc="text file containing the realignment parameters")


class FeatureTimeSeriesOutputSpec(TraitedSpec):
    maxRPcorr = traits.Array(desc="Array of the maximum RP correlation feature scores for the components of the melodic_mix file")


class FeatureTimeSeries(SimpleInterface):
    input_spec = FeatureTimeSeriesInputSpec
    output_spec = FeatureTimeSeriesOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        import random

        melmix = self.inputs.melmix
        mc = self.inputs.mc

        # Read melodic mix file (IC time-series), subsequently define a set of squared time-series
        mix = np.loadtxt(melmix)

        # Read motion parameter file
        rp6 = np.loadtxt(mc)
        _, nparams = rp6.shape

        # Determine the derivatives of the RPs (add zeros at time-point zero)
        rp6_der = np.vstack((np.zeros(nparams),
                            np.diff(rp6, axis=0)
                            ))

        # Create an RP-model including the RPs and its derivatives
        rp12 = np.hstack((rp6, rp6_der))

        # Add the squared RP-terms to the model
        # add the fw and bw shifted versions
        rp12_1fw = np.vstack((
            np.zeros(2 * nparams),
            rp12[:-1]
        ))
        rp12_1bw = np.vstack((
            rp12[1:],
            np.zeros(2 * nparams)
        ))
        rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

        # Determine the maximum correlation between RPs and IC time-series
        nsplits = 1000
        nmixrows, nmixcols = mix.shape
        nrows_to_choose = int(round(0.9 * nmixrows))

        # Max correlations for multiple splits of the dataset (for a robust estimate)
        max_correls = np.empty((nsplits, nmixcols))
        for i in range(nsplits):
            # Select a random subset of 90% of the dataset rows (*without* replacement)
            chosen_rows = random.sample(population=range(nmixrows),
                                        k=nrows_to_choose)

            # Combined correlations between RP and IC time-series, squared and non squared
            correl_nonsquared = cross_correlation(mix[chosen_rows],
                                                rp_model[chosen_rows])
            correl_squared = cross_correlation(mix[chosen_rows]**2,
                                            rp_model[chosen_rows]**2)
            correl_both = np.hstack((correl_squared, correl_nonsquared))

            # Maximum absolute temporal correlation for every IC
            max_correls[i] = np.abs(correl_both).max(axis=1)

        # Feature score is the mean of the maximum correlation over all the random splits
        # Avoid propagating occasional nans that arise in artificial test cases
        self._results['maxRPcorr'] = np.nanmean(max_correls, axis=0)
    
        return runtime


class FeatureFrequencyInputSpec(BaseInterfaceInputSpec):
    melFTmix = File(exists=True, desc="the melodic_FTmix text file")
    TR = traits.Float(desc="TR (in seconds) of the fMRI data")


class FeatureFrequencyOutputSpec(TraitedSpec):
    HFC = traits.Array(desc="Array of the HFC ('High-frequency content') feature scores for the components of the melodic_FTmix file")


class FeatureFrequency(SimpleInterface):
    input_spec = FeatureFrequencyInputSpec
    output_spec = FeatureFrequencyOutputSpec

    def _run_interface(self, runtime):
        import numpy as np

        # Determine sample frequency
        Fs = old_div(1, TR)

        # Determine Nyquist-frequency
        Ny = old_div(Fs, 2)

        # Load melodic_FTmix file
        FT = np.loadtxt(melFTmix)

        # Determine which frequencies are associated with every row in the melodic_FTmix file  (assuming the rows range from 0Hz to Nyquist)
        f = Ny * (np.array(list(range(1, FT.shape[0] + 1)))) / (FT.shape[0])

        # Only include frequencies higher than 0.01Hz
        fincl = np.squeeze(np.array(np.where(f > 0.01)))
        FT = FT[fincl, :]
        f = f[fincl]

        # Set frequency range to [0-1]
        f_norm = old_div((f - 0.01), (Ny - 0.01))

        # For every IC; get the cumulative sum as a fraction of the total sum
        fcumsum_fract = old_div(np.cumsum(FT, axis=0), np.sum(FT, axis=0))

        # Determine the index of the frequency with the fractional cumulative sum closest to 0.5
        idx_cutoff = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

        # Now get the fractions associated with those indices index, these are the final feature scores
        HFC = f_norm[idx_cutoff]

        # Return feature score
        self._results['HFC'] = HFC

        return runtime

class FeatureSpatialInputSpec(BaseInterfaceInputSpec):
    melodic_ic_file = File(exists=True, desc="nii.gz file containing mixture-modelled thresholded (p>0.5) Z-maps, registered to the MNI152 2mm template")
    aroma_dir = Directory(exists=True, desc="ICA-AROMA directory, containing the mask-files (mask_edge.nii.gz, mask_csf.nii.gz & mask_out.nii.gz)")


class FeatureSpatialOutputSpec(TraitedSpec):
    edge_fraction = traits.Array(desc="Array of the edge fraction feature scores for the components of the melIC file")
    csf_fraction = traits.Array(desc="Array of the CSF fraction feature scores for the components of the melIC file")


class FeatureSpatial(SimpleInterface):
    input_spec = FeatureSpatialInputSpec
    output_spec = FeatureSpatialOutputSpec

    def _run_interface(self, runtime):

        melodic_ic_file = self.inputs.melodic_ic_file
        aroma_dir = self.inputs.aroma_dir

        csf_mask  = join(aroma_dir, 'mask_csf.nii.gz')
        edge_mask = join(aroma_dir, 'mask_edge.nii.gz')
        out_mask  = join(aroma_dir, 'mask_out.nii.gz')

        total_sum, csf_sum, edge_sum, outside_sum = zsums(
            melodic_ic_file, masks=[None, csf_mask, edge_mask, out_mask]
        )

        edge_fraction = np.where(total_sum > csf_sum, (outside_sum + edge_sum) / (total_sum - csf_sum), 0)
        csf_fraction = np.where(total_sum > csf_sum, csf_sum / total_sum, 0)

        self._results['edge_fraction'] = edge_fraction
        self._results['csf_fraction'] = csf_fraction

        return runtime


class ClassificationInputSpec(BaseInterfaceInputSpec):
    maxRPcorr = traits.Array(desc="Array of the 'maximum RP correlation' feature scores of the components")
    edge_fraction = traits.Array(desc="Array of the 'edge fraction' feature scores of the components")
    HFC = traits.Array(desc="Array of the 'high-frequency content' feature scores of the components")
    csf_fraction = traits.Array(desc="Array of the 'CSF fraction' feature scores of the components")


class ClassificationOutputSpec(TraitedSpec):
    motionICs = traits.Array(desc="Array containing the indices of the components identified as motion components")
    classified_motion_ics = traits.File(desc="A text file containing the indices of the components identified as motion components")


class Classification(SimpleInterface):
    input_spec = ClassificationInputSpec
    output_spec = ClassificationOutputSpec

    def _run_interface(self, runtime):

        # Classify the ICs as motion or non-motion

        # Define criteria needed for classification (thresholds and hyperplane-parameters)
        thr_csf = 0.10
        thr_HFC = 0.35
        hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

        # Project edge & maxRPcorr feature scores to new 1D space
        x = np.array([maxRPcorr, edgeFract])
        proj = hyp[0] + np.dot(x.T, hyp[1:])

        # Classify the ICs
        motionICs = np.squeeze(np.array(np.where((proj > 0) + (csfFract > thr_csf) + (HFC > thr_HFC))))

        # Put the feature scores in a text file
        np.savetxt(os.path.join(outDir, 'feature_scores.txt'),
                np.vstack((maxRPcorr, edgeFract, HFC, csfFract)).T)

        # Put the indices of motion-classified ICs in a text file
        txt = open(os.path.join(outDir, 'classified_motion_ICs.txt'), 'w')
        if motionICs.size > 1:  # and len(motionICs) != 0: if motionICs is not None and 
            txt.write(','.join(['{:.0f}'.format(num) for num in (motionICs + 1)]))
        elif motionICs.size == 1:
            txt.write('{:.0f}'.format(motionICs + 1))
        txt.close()

        # Create a summary overview of the classification
        txt = open(os.path.join(outDir, 'classification_overview.txt'), 'w')
        txt.write('\t'.join(['IC',
                            'Motion/noise',
                            'maximum RP correlation',
                            'Edge-fraction',
                            'High-frequency content',
                            'CSF-fraction']))
        txt.write('\n')
        for i in range(0, len(csfFract)):
            if (proj[i] > 0) or (csfFract[i] > thr_csf) or (HFC[i] > thr_HFC):
                classif = "True"
            else:
                classif = "False"
            txt.write('\t'.join(['{:d}'.format(i + 1),
                                classif,
                                '{:.2f}'.format(maxRPcorr[i]),
                                '{:.2f}'.format(edgeFract[i]),
                                '{:.2f}'.format(HFC[i]),
                                '{:.2f}'.format(csfFract[i])]))
            txt.write('\n')
        txt.close()


def cross_correlation(a, b):
    """Cross Correlations between columns of two matrices"""
    assert a.ndim == b.ndim == 2
    _, ncols_a = a.shape
    # nb variables in columns rather than rows hence transpose
    # extract just the cross terms between cols in a and cols in b
    return np.corrcoef(a.T, b.T)[:ncols_a, ncols_a:]
