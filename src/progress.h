#include <stdio.h>
#include <sys/time.h>
#include <sys/ioctl.h>

// Determines the length of the progress bar. If your terminal is being overrun, try decreasing this.
#define BAR_LEN 20

void printProgress(struct timeval startTime, int d, int total, int freq)
{
   // Initialize timekeeping variables.
   float timeLeft;
   float dt = 0;
   int seconds, useconds;
   int min, sec, ms;
   int dMin, dSec, dMs;
   min = sec = ms = dMin = dSec = dMs = 0;

   // Set padding for strings to their length (minus one for null
   // terminating character) plus a specified value.
   int strPad = 3;
   int pad = 4;

   // Get terminal width.
   struct winsize w;
   ioctl(0, TIOCGWINSZ, &w);
   int termW = w.ws_col;

   // Length of time string.
   int timeLen = 8;
   // Length of percent string.
   int percentLen = 7;

   int maxBarLen = (pad * 2 + strPad * 2) + ((int)strlen("elapsed:") - 1)
      + ((int)strlen("eta:") - 1) + (timeLen * 2) + (percentLen + 1)
      + (BAR_LEN + 2) + 1;
   int midBarLen = (pad + (int)strlen("eta:") - 1 + strPad + timeLen
         + (percentLen + 1) + 1);
   int minBarLen = percentLen + 1;

   // bool fullProgressEnabled = maxBarLen > BAR_LEN;
   bool fullProgressEnabled = maxBarLen < termW;
   bool midProgressEnabled = midBarLen < termW;
   bool minProgressEnabled = minBarLen < termW;

   if (d % freq == 0 || d == total - 1)
   {
      // Get time.
      struct timeval curTime;
      gettimeofday(&curTime, NULL);
      seconds = (int)curTime.tv_sec - (int)startTime.tv_sec;
      useconds = (int)curTime.tv_usec - (int)startTime.tv_usec;
      dt = (float)(((seconds) * 1000 + useconds/1000.0) + 0.5);
      float percent = (float)(d + 1) / (float)total;

      timeLeft = ((float)dt / percent - (float)dt) / 1000.0f;

      // Calculate time data;
      min = (int)timeLeft / 60;
      sec = (int)timeLeft % 60;
      ms = (int)(timeLeft * 100) % 60;

      dMin = (int)(dt / 1000) / 60;
      dSec = (int)(dt / 1000) % 60;
      dMs = (int)(dt / 10) % 60;

      if (fullProgressEnabled)
      {
         // Print everything.
         string progress;
         // Fill progress bar.
         progress += "[";
         for (int j = 0; j < BAR_LEN; j++)
         {
            float j_percent = (float)j / (float)BAR_LEN;
            if (j_percent < percent)
            {
               progress += "=";
            }
            else
            {
               progress += "-";
            }
         }
         progress += "]";

         // Print data.
         printf("\r%s%*s%02d:%02d:%02d",
               "elapsed:", strPad, "", dMin, dSec, dMs);
         printf("%*s%s%*s%02d:%02d:%02d",
               pad, "", "eta:", strPad, "", min, sec, ms);
         // Display progress bar.
         printf("%*s%*.2f%% %s",
               pad, "", percentLen - 2, percent * 100.0f, progress.c_str());
      }
      else if (midProgressEnabled)
      {
         // Print the percent and the ETA.
         printf("\r%-*s %02d:%02d:%02d",
               (int)strlen("eta:") - 1 + strPad, "eta:", min, sec, ms);
         printf("%*s%.2f%%",
               pad, "", percent * 100.0f);
      }
      else if (minProgressEnabled)
      {
         // Print only the percent.
         printf("\r%.2f%%", percent * 100.0f);
      }
      /*
         else
         {
         printf("Warning: terminal must be at least %d characters wide. Data will not be displayed.\n", minBarLen);
         }
         printf("terminal width: %d (%d)", termW, maxBarLen);
         */

      // Flush stdout to print stats.
      fflush(stdout);
   }
}
