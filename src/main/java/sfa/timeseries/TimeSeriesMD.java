package sfa.timeseries;
import java.io.Serializable;

public class TimeSeriesMD implements Serializable {

    protected TimeSeries[] sources = null;
    protected int numSources = 0;
    protected String[] sourcesNames = null;
    protected String labelMD = null;

    public TimeSeriesMD(TimeSeries[] data) {
        this.sources = data;
        this.numSources = data.length;
    }

    public TimeSeriesMD(TimeSeries[] data, String label) {
        this(data);
        this.labelMD = label;
    }

    public TimeSeriesMD(TimeSeries[] data, String[] sourceNames, String label) {
        this(data, label);
        this.sourcesNames = sourceNames;
        this.numSources = sourceNames.length;
    }

    /**
     * The label for supervised data analytics
     *
     * @return
     */
    public String getLabel() {
        return this.labelMD;
    }

    public void setLabel(String label) {
        this.labelMD = label;
    }

    public int getLength() {
        if (this.numSources != 0) {
            return this.sources[0].getLength();
        } else {
            return 0;

        }
    }

    public int getNumSources() {
        return this.sources.length;

    }
    public TimeSeries getTimeSeriesOfOneSource(int idSource){
        if(idSource >= numSources || idSource < 0){
            return null;
        }else{
            return new TimeSeries(sources[idSource].data,this.getLabel());
        }
    }
}