package jsonbuilder.database;

public class Media {
    private String type;
    private String url;
    
    public Media(String type, String url){
        this.type = type;
        this.url = url;
    }

    public String getType() {
        return this.type;
    }

    public String getUrl() {
        return this.url;
    }
}
