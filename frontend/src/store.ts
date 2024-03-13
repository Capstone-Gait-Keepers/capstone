import { reactive } from 'vue'
import type { User } from '@/types'

export default reactive<{
  user: User | null;
  connected_users: string[];
  viewed_categories: {
    balance: boolean;
    neurodegenerative: boolean;
    dementia: boolean;
  };
}>({
  user: {
    name: "Marjorie",
    email: "marjorie.smith@gmail.com",
    password: "thisisapassword",
    sensorid: 39,
  }, // TODO: Default should be null
  connected_users: ['dan@raymond.ch'],
  viewed_categories: {
    balance: true,
    neurodegenerative: true,
    dementia: true,
  }
});
