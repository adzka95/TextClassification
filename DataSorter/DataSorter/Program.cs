using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataSorter
{
    class Program
    {
        static void Main(string[] args)
        {
            string filePath = "C:\\Users\\Ada\\Desktop\\TextClassification\\Dane\\Surowe\\Part1";
            string destinationPath = "C:\\Users\\Ada\\Desktop\\TextClassification\\Dane\\Posortowane";

            var listOfCategories = new Dictionary<string, long>();

            string[] allFiles = Directory.GetFiles(filePath, "*.txt*", SearchOption.AllDirectories);
            foreach (var file in allFiles)
            {
                FileInfo info = new FileInfo(file);
                bool writeToFile = true;
                String strLine, fileDestinationPath ="";
                var fileContent = new System.Text.StringBuilder();
                using (StreamReader streamReader = new StreamReader(info.FullName))
                {
                    while (!streamReader.EndOfStream)
                    {
                        strLine = streamReader.ReadLine();
                        if (strLine.Contains("Fld Applictn"))
                        {
                            writeToFile = false;
                            strLine = strLine.Replace("Fld Applictn: ", "");
                            strLine = strLine.Replace("/", "-");
                            if (strLine.IndexOf(" ") > 0)
                            {
                                strLine = strLine.Remove(0, strLine.IndexOf(" ")).Trim();
                            }
                            strLine = strLine.Trim();
                            string category = strLine.Replace(" ", "_");

                            if (category == "")
                            {
                                fileDestinationPath = "\\undefined";
                            }
                            else
                            {
                                long categoryIndex = 0;                                
                                if (listOfCategories.ContainsKey(category))
                                {
                                    categoryIndex= listOfCategories[category]++;
                                }
                                else
                                {
                                    listOfCategories.Add(category, 0);
                                }
                                switch (categoryIndex % 3) {
                                    case 0:
                                        fileDestinationPath = "\\train" ;
                                        break;
                                    case 1:
                                        fileDestinationPath = "\\test";
                                        break;
                                    case 2:
                                        fileDestinationPath = "\\validate";
                                        break;
                                }
                                fileDestinationPath += "\\" + category;
                            }
                        }
                        if (strLine.Contains("Program Ref")) {
                            writeToFile = true;
                        }
                        if (writeToFile) {
                            fileContent.AppendLine(strLine);
                        }
                    }
                }
                string finalDestination = destinationPath + fileDestinationPath;
                //Console.WriteLine(finalDestination);
                bool exists = System.IO.Directory.Exists(finalDestination);
                if (!exists)
                {
                    System.IO.Directory.CreateDirectory(finalDestination);
                }
                System.IO.StreamWriter fileWriter = new System.IO.StreamWriter(finalDestination + "\\" + info.Name);
                fileWriter.WriteLine(fileContent.ToString()); 
            }
            Console.WriteLine("Work is done");
            Console.Read();
        }
    }
}
