package sfa.timeseries;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.SFAWordsTest;
import sfa.classification.*;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.File;
import java.io.IOException;

@RunWith(JUnit4.class)
public class TimeSeriesLoaderTest {

    //The dataset to use
    public static String[] datasets = new String[]{
            "WISDM-MDU",
    };

    @Test
    public void loadMultivariateDatasetTestMethod() throws IOException {
        // the relative path to the datasets
        ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
        File dir = new File(classLoader.getResource("datasets/").getFile());

        for (String s : datasets) {

            File d = new File(dir.getAbsolutePath() + "/" + s);
            if (d.exists() && d.isDirectory()) {

                int num_sources = 3;
                int segment_length = 200;
                String filename = dir.getAbsolutePath() + "/" + s + "/" + "user1";
                File train = new File(filename);


                MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDataset(train, num_sources, segment_length);

                int tamanho = trainSamples.length;
               // System.out.println("Done reading from " + filename + " samples " + tamanho + " queryLength " + segment_length);

            } else {
                System.err.println("Does not exist!" + d.getAbsolutePath());
            }
        }

    }


}
