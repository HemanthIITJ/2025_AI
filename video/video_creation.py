import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
import numpy as np
from scipy.io.wavfile import write

class VideoCreator:
    """
    A class to create high-quality videos from lists of images, text, and audio.
    """
    VIDEO_CODEC = 'libx264'
    VIDEO_QUALITY = 'hq'  # 'hq' or 'lq' for high or low quality, adjust parameters accordingly

    def __init__(self, image_files, text_contents, audio_files, background_image_path=None, font_path=None):
        """
        Initializes VideoCreator with lists of image files, text contents, and audio files.

        :param image_files: List of image file paths.
        :param text_contents: List of text strings to overlay.
        :param audio_files: List of audio file paths (will be concatenated).
        :param background_image_path: Path to background image (optional).
        :param font_path: Path to font file for text overlays (optional).
        """
        self.image_files = self._validate_files(image_files, 'image')
        self.text_contents = text_contents if text_contents else []
        self.audio_files = self._validate_files(audio_files, 'audio')
        self.background_image_path = self._validate_file(background_image_path, 'background image', allow_none=True)
        self.font_path = self._validate_file(font_path, 'font', allow_none=True) if font_path else None
        self.video_size = None  # Will be set by user choice
        self.output_video_path = 'output_video.mp4'  # Default output path

    def _validate_files(self, file_paths, file_type):
        """
        Validates if file paths are valid and files exist.
        """
        if not file_paths:
            return []  # Allow empty lists

        validated_files = []
        for file_path in file_paths:
            validated_path = self._validate_file(file_path, file_type)
            validated_files.append(validated_path)
        return validated_files

    def _validate_file(self, file_path, file_type, allow_none=False):
        """
        Validates single file path.
        """
        if allow_none and file_path is None:
            return None

        if not file_path:
            raise FileNotFoundError(f"Empty {file_type} file path provided.")
        
        # Resolve absolute path and check existence
        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"{file_type.capitalize()} file not found: {abs_path}")
        return abs_path
    
    def get_output_video_path(self):
       """Returns output path set by the method"""
       return self.output_video_path

    def _process_images(self):
        """
        Loads and processes images with improved path handling.
        """
        frames = []
        bg_image = None
        if self.background_image_path:
            bg_image = cv2.imread(self.background_image_path)
            if bg_image is None:
                raise Exception(f"Failed to load background image: {self.background_image_path}")
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

        for image_file in self.image_files:
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"Warning: Could not read image file: {image_file}. Skipping.")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if bg_image is not None:
                frame = self._overlay_image_on_background(frame, bg_image)

            if self.video_size:
                frame = self._resize_frame(frame, self.video_size)

            frames.append(frame)
        return frames

    def _overlay_image_on_background(self, frame, background):
        """
        Overlays the image frame onto the background.
        """
        h_frame, w_frame = frame.shape[:2]
        h_bg, w_bg = background.shape[:2]

        # Resize frame to fit within the background if necessary
        if w_frame > w_bg or h_frame > h_bg:
            scale = min(w_bg / w_frame, h_bg / h_frame)
            new_w = int(w_frame * scale)
            new_h = int(h_frame * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h_frame, w_frame = new_h, new_w
        
        x_offset = (w_bg - w_frame) // 2
        y_offset = (h_bg - h_frame) // 2

        background = background.copy()  # Avoid modifying original
        background[y_offset:y_offset+h_frame, x_offset:x_offset+w_frame] = frame
        return background

    def _resize_frame(self, frame, size):
        """Resizes the frame to fit the video size."""
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


    def _get_video_size(self, video_size_choice):
       """
        Determines video size based on user choice.
       """
       if video_size_choice == 'HD':
          return (1280, 720)
       elif video_size_choice == 'Full HD':
          return (1920, 1080)
       elif video_size_choice == '4K':
          return (3840, 2160)
       else:
          # Default to HD if choice is not valid
          return (1280,720)

    def _process_text_frames(self, frame_size, duration_per_text=2):
        """
        Generates text frames with configurable font path.
        """
        text_frames = []
        frame_duration_list = []
        if not self.text_contents:
            return [], []

        width, height = frame_size
        font_size = int(min(width, height) / 15)
        
        try:
            # Try user-specified font first, then system default
            if self.font_path:
                font = ImageFont.truetype(self.font_path, font_size)
            else:
                # Attempt to find system default font
                font = ImageFont.truetype(ImageFont.findfont(ImageFont.load_default()), font_size)
        except IOError:
            print("Warning: Specified font not found, using default font")
            font = ImageFont.load_default()

        for text in self.text_contents:
            img = Image.new('RGB', frame_size, color='white')
            d = ImageDraw.Draw(img)
            
            # Improved text positioning with multiline support
            text_lines = text.split('\n')
            
            # Use textbbox to get precise bounding boxes for text
            line_bboxes = [d.textbbox((0, 0), line, font=font) for line in text_lines]
            line_heights = [bbox[3] - bbox[1] for bbox in line_bboxes] # Extract height
            total_height = sum(line_heights)
            
            y = (height - total_height) // 2
            for line, line_height, bbox in zip(text_lines, line_heights, line_bboxes):
                text_width = bbox[2] - bbox[0]  # width = x2 - x1
                x = (width - text_width) // 2
                d.text((x, y), line, fill=(0, 0, 0), font=font)
                y += line_height

            text_frame_np = np.array(img)
            text_frames.append(text_frame_np)
            frame_duration_list.append(duration_per_text)

        return text_frames, frame_duration_list

    def _process_audio(self):
        """
        Processes and combines audio files with error handling.
        """
        if not self.audio_files:
            return None

        audio_clips = []
        for audio_file in self.audio_files:
            try:
                clip = AudioFileClip(audio_file)
                audio_clips.append(clip)
            except Exception as e:
                print(f"Warning: Could not load audio file {audio_file}: {str(e)}")

        if not audio_clips:
            return None

        # Handle different audio formats and sample rates
        return CompositeAudioClip(audio_clips)

    def create_video(self, output_path='output_video.mp4', fps=24, video_size_choice='Full HD'):
        """
        Creates the video with comprehensive error handling and proper clip concatenation.
        """
        try:
            # Set up output path with directory creation
            self.output_video_path = os.path.abspath(output_path)
            output_dir = os.path.dirname(self.output_video_path)
            os.makedirs(output_dir, exist_ok=True)

            # Determine video size
            self.video_size = self._get_video_size(video_size_choice)
            
            # Process media components
            image_frames = self._process_images()
            text_frames, text_durations = self._process_text_frames(self.video_size)
            audio_clip = self._process_audio()

            # Combine all frames with durations
            all_frames = []
            frame_durations = []
            
            # Add image frames with default duration
            image_frame_duration = 3  # seconds per image
            for frame in image_frames:
                all_frames.append(frame)
                frame_durations.append(image_frame_duration)
            
            # Add text frames with specified durations
            for frame, duration in zip(text_frames, text_durations):
                all_frames.append(frame)
                frame_durations.append(duration)

            if not all_frames:
                raise ValueError("No valid frames available for video creation")

            # Create video clips with proper durations
            clips = []
            for frame, duration in zip(all_frames, frame_durations):
                clip = ImageSequenceClip([frame], durations=[duration])
                clips.append(clip)

            # Concatenate all clips temporally
            final_clip = concatenate_videoclips(clips, method="compose")

            # Add audio if available
            if audio_clip:
                # Match audio duration to video duration
                if audio_clip.duration > final_clip.duration:
                    audio_clip = audio_clip.subclip(0, final_clip.duration)
                # final_clip = final_clip.set_audio(audio_clip)

            # Video encoding parameters
            video_params = {
                'codec': self.VIDEO_CODEC,
                'fps': fps,
                'preset': 'medium',  # Balanced encoding speed/quality
                'audio_codec': 'aac',
                'threads': 4,  # Multi-threading for faster encoding
                'logger': None  # Disable verbose moviepy output
            }

            # Write final video file
            final_clip.write_videofile(
                self.output_video_path,
                **video_params
            )

            # Cleanup resources
            final_clip.close()
            if audio_clip:
                audio_clip.close()

            print(f"Successfully created video: {self.output_video_path}")
            return True

        except Exception as e:
            print(f"Error creating video: {str(e)}")
            return False

    def __del__(self):
        """Cleanup resources when instance is destroyed"""
        pass  # MoviePy clips should already be closed, but could add additional cleanup

# Example usage
if __name__ == "__main__":
    # Example file paths (use absolute paths for reliability)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create dummy folders and files
    os.makedirs(os.path.join(current_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'assets'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'fonts'), exist_ok=True)
    os.makedirs(os.path.join(current_dir, 'output'), exist_ok=True)


    # Create a dummy image file
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image[:] = (255, 0, 0)  # Red color
    cv2.imwrite(os.path.join(current_dir, 'images', 'image1.jpg'), dummy_image)

    dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_image[:] = (0, 255, 0) # Green
    cv2.imwrite(os.path.join(current_dir, 'images', 'image2.png'), dummy_image)
    
    # Create a dummy audio file 
    # Create a dummy 1 second audio file using sine wave
    sample_rate = 44100
    duration = 1
    frequency = 440  # Hz

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Convert to int16 data (common format for audio files)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    
    write(os.path.join(current_dir, 'audio', 'background_music.wav'), sample_rate, audio_data_int16)

    # Create a dummy background image
    bg_image = np.zeros((300, 300, 3), dtype=np.uint8)
    bg_image[:] = (0, 0, 255) # blue
    cv2.imwrite(os.path.join(current_dir, 'assets', 'bg.jpg'), bg_image)
    
    # Create a dummy font file
    # Here we create a dummy font file using PIL (it's not a true .ttf file, but for testing we can use a default one)
    from PIL import ImageFont
    default_font = ImageFont.load_default()
    # Save the font
    with open(os.path.join(current_dir, 'fonts', 'arial.ttf'), "w") as f:
      f.write("Dummy font") #just a placeholder

    image_files = [
        os.path.join(current_dir, 'images', 'image1.jpg'),
        os.path.join(current_dir, 'images', 'image2.png')
    ]
    
    audio_files = [
        os.path.join(current_dir, 'audio', 'background_music.wav')
    ]
    
    text_contents = [
        "Welcome to Our Presentation\nVersion 1.0",
        "Conclusion\nThank You!"
    ]
    
    creator = VideoCreator(
        image_files=image_files,
        text_contents=text_contents,
        audio_files=audio_files,
        background_image_path=os.path.join(current_dir, 'assets', 'bg.jpg'),
        font_path=os.path.join(current_dir, 'fonts', 'arial.ttf')
    )
    
    success = creator.create_video(
        output_path=os.path.join(current_dir, 'output', 'presentation.mp4'),
        fps=30,
        video_size_choice='Full HD'
    )
    
    if success:
        print(f"Output video created at: {creator.get_output_video_path()}")
    
    # Remove dummy files and folders after video creation
    os.remove(os.path.join(current_dir, 'images', 'image1.jpg'))
    os.remove(os.path.join(current_dir, 'images', 'image2.png'))
    os.remove(os.path.join(current_dir, 'audio', 'background_music.wav'))
    os.remove(os.path.join(current_dir, 'assets', 'bg.jpg'))
    os.remove(os.path.join(current_dir, 'fonts', 'arial.ttf'))
    
    os.rmdir(os.path.join(current_dir, 'images'))
    os.rmdir(os.path.join(current_dir, 'audio'))
    os.rmdir(os.path.join(current_dir, 'assets'))
    os.rmdir(os.path.join(current_dir, 'fonts'))
    # os.rmdir(os.path.join(current_dir, 'output')) # output might be necessary if the user wants to run code multiple times