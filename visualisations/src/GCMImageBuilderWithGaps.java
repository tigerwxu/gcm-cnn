import java.awt.image.BufferedImage;
import java.io.*;
import javax.imageio.*;
public class GCMImageBuilderWithGaps
{
	static final int HEIGHT = 131;
	static final int WIDTH = 360;
	static double MIN, MAX;
	static final String FILEPATH = "C:\\Users\\tiger\\Desktop\\COMPX591\\Data\\SI_archive\\SST_1993_2016_ecmwf.csv";
	
	
	public static void main(String[] args) throws Exception
	{
		findRange();

		BufferedReader in = new BufferedReader(new FileReader(FILEPATH));
		BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
		
		//first line needs to be ignored
		in.readLine();
		
		//2nd line is longitude
		String[] longitudes = in.readLine().split(",");
		
		//3rd line is latitude
		String[] latitudes = in.readLine().split(",");
		
		//succeeding lines will describe global snapshots with time as the 0th entry
		
		
		String line;
		while ((line = in.readLine()) != null)
		{
			String[] currentLine = line.split(",");
			for (int i = 1; i < longitudes.length; i++)
			{
				int x = (int) Double.parseDouble(longitudes[i]);
				int y = 40 - ((int) Double.parseDouble(latitudes[i]));
				int intensity = normalisetoRGB(Double.parseDouble(currentLine[i]));
				img.setRGB(x, y, intensity);
				
			}
			
			ImageIO.write(img, "gif", new File(currentLine[0] + ".gif"));
		}
		
		
		
		
		in.close();
	}
	public static int normalisetoRGB(double val)
	{
		double range = MAX - MIN;
		double normalised = (val - MIN) / range;
		int intensity = (int) ((normalised * 255) + 0.5); //equivalent to rounding to the nearest integer (+0.5 then truncate)
		return ((intensity << 16) | (intensity << 8) | intensity);
	}
	
	//finds the maximum and minimum in the entire data-set
	public static void findRange() throws Exception
	{
		
		BufferedReader in = new BufferedReader(new FileReader(FILEPATH));
		in.readLine();
		in.readLine();
		in.readLine();
		MIN = Double.POSITIVE_INFINITY;
		
		String line;
		while ((line = in.readLine()) != null)
		{
			String[] splitLine = line.split(",");
			for (int i = 1; i < splitLine.length; i++) //start i = 1 as 0th entry gives date
			{
				double val = Double.parseDouble(splitLine[i]);
				if (val < MIN)
				{
					MIN = val;
				}
				else if (val > MAX)
				{
					MAX = val;
				}
			}
		}
		
		in.close();
		
	}

}
