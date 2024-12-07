import localforage from 'localforage';

export const offlineStorage = {
  async setItem(key: string, value: any) {
    try {
      await localforage.setItem(key, value);
    } catch (error) {
      console.error('Error setting offline data:', error);
    }
  },

  async getItem(key: string) {
    try {
      return await localforage.getItem(key);
    } catch (error) {
      console.error('Error getting offline data:', error);
      return null;
    }
  },

  async removeItem(key: string) {
    try {
      await localforage.removeItem(key);
    } catch (error) {
      console.error('Error removing offline data:', error);
    }
  },

  async clear() {
    try {
      await localforage.clear();
    } catch (error) {
      console.error('Error clearing offline data:', error);
    }
  },
};

