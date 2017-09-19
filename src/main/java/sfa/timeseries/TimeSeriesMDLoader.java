package sfa.timeseries;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.carrotsearch.hppc.DoubleArrayList;

public class TimeSeriesMDLoader {

    public static TimeSeriesMD[] loadDatset(File[] datasets, File labels) throws IOException {
        int numOfInstancias = 0;
        int numSources = datasets.length;
        ArrayList<String> labelsForEachWindowResult = new ArrayList<String>();
        ArrayList<TimeSeriesMD> instancias = new ArrayList<TimeSeriesMD>();

        try (BufferedReader br = new BufferedReader(new FileReader(labels))) {
            String line = null;
            String[] labelsForEachWindow = null;

            while ((line = br.readLine()) != null) {
                labelsForEachWindow = line.split(",");
                int j = 0;

                for (int i = 0; i < labelsForEachWindow.length; i++) {
                    String column = labelsForEachWindow[i].trim();
                    try {
                        if (isNonEmptyColumn(column)) {
                            labelsForEachWindowResult.add(column);
                        }
                    } catch (NumberFormatException nfe) {
                        nfe.printStackTrace();
                    }
                }
                numOfInstancias = labelsForEachWindowResult.size();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Signals
        BufferedReader[] readers = new BufferedReader[numSources];

        for (int sourceId = 0; sourceId < numSources; sourceId++) {

            readers[sourceId] = new BufferedReader(new FileReader(datasets[sourceId]));

        }

        try {
            for (int iLine = 0; iLine < numOfInstancias; iLine++) {
                double[][] lines = new double[numSources][];
                for (int ireaders = 0; ireaders < numSources; ireaders++) {
                    String line = readers[ireaders].readLine();
                    if (line != null) {
                        String[] columns = line.split(",");
                        double[] dataOfOneLine = new double[columns.length];
                        int j = 0;
                        for (int i = 0; i < columns.length; i++) {
                            String column = columns[i].trim();
                            try {
                                if (isNonEmptyColumn(column)) {
                                    dataOfOneLine[j++] = Double.parseDouble(column);
                                }
                            } catch (NumberFormatException nfe) {
                                nfe.printStackTrace();
                            }
                        }
                        lines[ireaders] = dataOfOneLine;
                    }
                }
                TimeSeries[] tsdata = new TimeSeries[numSources];
                for (int i = 0; i < numSources; i++) {
                    tsdata[i] = new TimeSeries(Arrays.copyOfRange(lines[i], 0, lines[i].length));
                }

                TimeSeriesMD tsMD = new TimeSeriesMD(tsdata, labelsForEachWindowResult.get(iLine));
                instancias.add(tsMD);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        //System.out.println("Done reading from " + dataset + " samples " + samples.size() + " length " + samples.get(0).getLength());
        return instancias.toArray(new TimeSeriesMD[]{});
    }

    public static TimeSeries readSampleSubsequence(File dataset) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(dataset))) {
            DoubleArrayList data = new DoubleArrayList();
            String line = null;
            while ((line = br.readLine()) != null) {
                line.trim();
                String[] values = line.split("[ \\t]");
                if (values.length > 0) {
                    for (String value : values) {
                        try {
                            value.trim();
                            if (isNonEmptyColumn(value)) {
                                data.add(Double.parseDouble(value));
                            }
                        } catch (NumberFormatException nfe) {
                            // Parse-Exception ignorieren
                        }
                    }
                }
            }
            return new TimeSeries(data.toArray());
        }
    }

    public static TimeSeries[] readSamplesQuerySeries(File dataset) throws IOException {
        List<TimeSeries> samples = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(dataset))) {
            String line = null;
            while ((line = br.readLine()) != null) {
                DoubleArrayList data = new DoubleArrayList();
                line.trim();
                String[] values = line.split("[ \\t]");
                if (values.length > 0) {
                    for (String value : values) {
                        try {
                            value.trim();
                            if (isNonEmptyColumn(value)) {
                                data.add(Double.parseDouble(value));
                            }
                        } catch (NumberFormatException nfe) {
                            // Parse-Exception ignorieren
                        }
                    }
                    samples.add(new TimeSeries(data.toArray()));
                }
            }
        }
        return samples.toArray(new TimeSeries[]{});
    }

    public static boolean isNonEmptyColumn(String column) {
        return column != null && !"".equals(column) && !"NaN".equals(column) && !"\t".equals(column);
    }

    public static TimeSeries generateRandomWalkData(int maxDimension, Random generator) {
        double[] data = new double[maxDimension];

        // Gaussian Distribution
        data[0] = generator.nextGaussian();

        for (int d = 1; d < maxDimension; d++) {
            data[d] = data[d - 1] + generator.nextGaussian();
        }

        return new TimeSeries(data);
    }
}
