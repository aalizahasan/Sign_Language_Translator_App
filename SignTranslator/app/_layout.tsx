import { Tabs } from 'expo-router';
import React from 'react';

export default function Layout() {
  return (
    <Tabs>
      <Tabs.Screen name="camera" options={{ title: "Camera" }} />
      <Tabs.Screen name="index" options={{ title: "Home" }} />
    </Tabs>
  );
}
