package jsonbuilder.database;
import java.sql.*;

import java.util.ArrayList;

public class DataMgr {
    private static final String URL = "jdbc:mysql://localhost:3306/StudentPanel";
    private static String USERNAME;
    private static String PASSWORD;
    Connection connection;

    public DataMgr(String username, String password) {
        USERNAME = username;
        PASSWORD = password;
    }

    public Connection getConnection() throws SQLException {
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
        } catch (ClassNotFoundException e) {
            throw new SQLException("MySQL Driver not found", e);
        }
        this.connection = DriverManager.getConnection(URL, USERNAME, PASSWORD);
        return this.connection;
    }

    public String getTag(int myIntentId) {
        String tag = "";
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getTag(?)}");
            stmt.setInt(1, myIntentId);
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                tag = rs.getString("Tag");
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return tag;
    }

    public String getAnswer(int myIntentId) {
        String answer = "";
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getAnswer(?)}");
            stmt.setInt(1, myIntentId);
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                answer = rs.getString("Answer");
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return answer;
    }
    
    public ArrayList<String> getMedia(int myIntentId) {
        ArrayList<String> media = new ArrayList<>();
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getMedia(?)}");
            stmt.setInt(1, myIntentId);
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                media.add(rs.getString("Type"));
                media.add(rs.getString("URL"));
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return media;
    }

    public ArrayList<String> getKeywords(int myIntentId) {
        ArrayList<String> keywords = new ArrayList<String>();
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getKeywords(?)}");
            stmt.setInt(1, myIntentId);
            ResultSet rs = stmt.executeQuery();
            while (rs.next()) {
                keywords.add(rs.getString("Keywords"));
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return keywords;
    }

    public int getNumberOfIntents() {

        int number = 0;
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getNumberOfIntents()}");
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                number = Integer.parseInt(rs.getString("Number_Of_Intents"));
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return number;
    }

}