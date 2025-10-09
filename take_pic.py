import subprocess

def take_pic( filename ):
 subprocess.run( [ "fswebcam", "-d", "/dev/video0", "-r", "4056x3040", filename ] )
 print( f"Photo saved to { filename }" )


if __name__ == "__main__":
 take_pic( "./input/latest.jpg" )

