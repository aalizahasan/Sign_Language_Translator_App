import { View, Text, Button, StyleSheet } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Video, ResizeMode } from 'expo-av';
import { useState } from 'react';

export default function HomeScreen() {
  const [video, setVideo] = useState<string | null>(null);
  const [result, setResult] = useState<string>('');

  // Camera se video capture
  const openCamera = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      alert('Camera permission required');
      return;
    }

    const capturedVideo = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });

    if (!capturedVideo.canceled) {
      const uri = capturedVideo.assets[0].uri;
      setVideo(uri);
      sendToAI(uri);
    }
  };

const sendToAI = async (uri: string) => {
  try {
    const filename = uri.split('/').pop() || 'video.mp4';

    const formData = new FormData();
    formData.append("file", {
      uri,
      name: filename,
      type: "video/mp4"
    } as any);

    console.log("Uploading:", uri);

    const response = await fetch("http://10.134.243.229:8000/predict-video", {
      method: "POST",
      body: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    const data = await response.json();
    console.log("Backend:", data);

    setResult(data.prediction || "No prediction");
  } catch (err) {
    console.log("Error sending video:", err);
  }
};


  return (
    <View style={styles.container}>
      <Text style={styles.title}>Sign Language Translator</Text>

      <Button title="Open Camera" onPress={openCamera} />

      {video && (
        <Video
          source={{ uri: video }}
          useNativeControls
          style={styles.video}
          resizeMode={ResizeMode.CONTAIN}
        />
      )}

      {result !== '' && <Text style={styles.result}>Detected: {result}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 22,
    marginBottom: 20,
  },
  video: {
    width: 250,
    height: 250,
    marginTop: 20,
  },
  result: {
    marginTop: 15,
    fontSize: 18,
    color: 'green',
  },
});
