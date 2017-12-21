package sfa.SubjectDependent;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.SFAWordsTest;
import sfa.classification.BOSSMDStackClassifier;
import sfa.classification.Classifier;
import sfa.classification.ParallelFor;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Locale;

@RunWith(JUnit4.class)
public class MainSubjectDependent {

    // The multivariate datasets to use
    public static String[] datasets = new String[]{
            //"WISDM-MDU",
            //"UCI-MDU",
            "UniMiB-MDU",
    };

    @Test
    public void testSubjectDependentClassification() throws IOException {
        try {
            // the relative path to the datasets

            boolean DEBUG = false;
            ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

            File dir = new File(classLoader.getResource("datasets/").getFile());

            for (String s : datasets) {
                File d = new File(dir.getAbsolutePath() + "/" + s);
                System.out.println("dataset,userId,numSources,samples,accuracy,precision,recall,f-measure");
                if (d.exists() && d.isDirectory()) {


                    for (File train : d.listFiles()) {

                        int num_sources = 3;
                        int segment_length = 151;
                        String filename = train.getName();
                        Classifier.DEBUG = DEBUG;
                        TimeSeriesLoader.DEBUG = DEBUG;
                        MultiVariateTimeSeries[] allSamples = TimeSeriesLoader.loadMultivariateDataset(train, num_sources, segment_length);

                        int folds = 10;
                        int[][] testIndices = new int[folds][];
                        int[][] trainIndices = new int[folds][];

                        Classifier.generateIndicesStatic(allSamples, folds,trainIndices,testIndices);

                        double somaAccuracy = 0;
                        double somaPrecision = 0;
                        double somaRecall= 0;
                        double somafmeasure = 0;

                        for (int f = 0; f < folds; f++){

                            MultiVariateTimeSeries[] trainSamples = getSamplesUsingIndes(allSamples,trainIndices[f]);
                            MultiVariateTimeSeries[] testSamples = getSamplesUsingIndes(allSamples,testIndices[f]);
                            BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                            Classifier.Score result = stack.eval(trainSamples,testSamples);

                            somaAccuracy += result.confusionMatrix.getAccuracy();
                            somaPrecision += result.confusionMatrix.getAvgPrecision();
                            somaRecall += result.confusionMatrix.getAvgRecall();
                            somafmeasure += result.confusionMatrix.getMacroFMeasure();
                            if(DEBUG) {
                                System.out.println(result.outputString);
                            }

                        }
                        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
                        otherSymbols.setDecimalSeparator('.');
                        DecimalFormat df = new DecimalFormat("#.00",otherSymbols);
                        System.out.println(s + "," + filename + "," + num_sources + "," + allSamples.length + "," + df.format(somaAccuracy/ folds * 100) + "," + df.format(somaPrecision/ folds * 100) + "," + df.format(somaRecall/ folds * 100) + "," + df.format(somafmeasure/ folds * 100));
                    }
                }

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

    private MultiVariateTimeSeries[] getSamplesUsingIndes(MultiVariateTimeSeries[] allSamples, int[] indices) {
        MultiVariateTimeSeries[] result = new MultiVariateTimeSeries[indices.length];
        int count = 0;
        for (int i : indices) {
            result[count] = allSamples[i];
            count++;
        }
        return result;
    }

}
