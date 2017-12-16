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

@RunWith(JUnit4.class)
public class MainSubjectDependent {

    // The multivariate datasets to use
    public static String[] datasets = new String[]{
            "WISDM-MDU",
    };

    @Test
    public void testSubjectDependentClassification() throws IOException {
        try {
            // the relative path to the datasets
            ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

            File dir = new File(classLoader.getResource("datasets/").getFile());

            for (String s : datasets) {
                File d = new File(dir.getAbsolutePath() + "/" + s);

                if (d.exists() && d.isDirectory()) {
                    for (File train : d.listFiles()) {

                        int num_sources = 3;
                        int segment_length = 200;
                        String filename = train.getName();

                        MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDataset(train, num_sources, segment_length);
                        BOSSMDStackClassifier stack = new BOSSMDStackClassifier();
                        stack.evalCrossValidation(trainSamples);
                        //Classifier.DEBUG = true;

                        //boolean useDerivatives = true;

                        System.out.println(s + ";" + filename + ";");
                    }
                }

                System.out.println("==============");
                System.out.println("= " + d.getName() + " =");
                System.out.println("======FIM=====");

            }
        } finally {
            ParallelFor.shutdown();
        }
    }

}
