import java.sql.*;

import java.util.ArrayList;
import java.util.List;

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

    public String getQuestion(int myQuestionId) {
        String question = "";
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getQuestion(?)}");
            stmt.setInt(1, myQuestionId);
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                question = rs.getString("Question");
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return question;
    }

    public String getAnswer(int myQuestionId) {
        String answer = "";
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getAnswer(?)}");
            stmt.setInt(1, myQuestionId);
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                answer = rs.getString("Answer");
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return answer;
    }

    public ArrayList<String> getPatterns(int myQuestionId) {
        ArrayList<String> patterns = new ArrayList<String>();
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getPatterns(?)}");
            stmt.setInt(1, myQuestionId);
            ResultSet rs = stmt.executeQuery();
            while (rs.next()) {
                patterns.add(rs.getString("Patterns"));
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return patterns;
    }

    public int getNumberOfQuestions() {
        int number = 0;
        try {
            CallableStatement stmt = this.connection.prepareCall("{CALL getNumberOfQuestions()}");
            ResultSet rs = stmt.executeQuery();
            if (rs.next()) {
                number = Integer.parseInt(rs.getString("Number_Of_Questions"));
            }
        } catch (SQLException e) {
            System.out.println("Failed to execute stored procedure: " + e.getMessage());
        }
        return number;
    }
}