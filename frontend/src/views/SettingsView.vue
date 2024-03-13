<template>
  <BasePage class="content" v-if="user">
    <h1>Account Settings</h1>
    <h2>Account Information</h2>
    <FormInput label="Name" type="text" v-model="user.name" />
    <FormInput label="Email" type="email" v-model="user.email" />
    <FormInput label="Password" type="password" v-model="user.password" />
    <FormInput label="Sensor ID" type="text" v-model="user.sensorid" />
    <!-- <FormInput label="Notifications" type="text" v-model="user.notifications" /> -->
    <h2>Metrics</h2>
    <p>Choose what information to view on the Home Insights page. If you don't want to see a specific group of insights or recommendations simply turn the toggle off. The system will still collect the data it just won’t display it to you so you can turn them back on at any time if you decide you want to see the insights.</p>
    <FormSwitch label="Balance indicators" id="balance" v-model="viewed_categories.balance" />
    <FormSwitch label="Neurodegenerative disease indicators" id="neurodegenerative" v-model="viewed_categories.neurodegenerative" />
    <FormSwitch label="Dementia indicators" id="dementia" v-model="viewed_categories.dementia" />
    <h2>Connected Users</h2>
    <i>Adding family members will allow them to view your individual insights from their own account. Once you add a family member they will receive an email inviting them to create an account and link your data to it. </i>
    <br>
    <div v-for="user_email, i in connected_users" :key="user_email" class="connected-user">
      <p>{{ user_email }}</p>
      <button @click="connected_users.splice(i, 1)">Delete</button>
    </div>
    <FormInput v-if="addingUser" id="new-user" type="email"  v-on:save="addUser" startEditing />
    <p v-if="addedUser" class="success">
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path fill-rule="evenodd" clip-rule="evenodd" d="M14 0C6.26801 0 0 6.26801 0 14C0 21.732 6.26801 28 14 28C21.732 28 28 21.732 28 14C28 10.287 26.525 6.72601 23.8995 4.10051C21.274 1.475 17.713 0 14 0ZM12.12 19.3797L7.12 14.4197L8.71 12.8497L12.12 16.2297L19.53 8.87969L21.12 10.4597L12.12 19.3797ZM2 14C2 20.6274 7.37258 26 14 26C20.6274 26 26 20.6274 26 14C26 7.37258 20.6274 2 14 2C7.37258 2 2 7.37258 2 14Z" fill="#68AE21"/>
      </svg>
      {{ addedUser }} successfully added as an Authorized User.
    </p>
    <button @click="startAddingUser">Add Family Member</button>
  </BasePage>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import BasePage from '@/components/BasePage.vue';
import FormInput from '@/components/FormInput.vue';
import FormSwitch from '@/components/FormSwitch.vue';
import store from '@/store';
const { viewed_categories, user, connected_users } = store;

const addingUser = ref(false);
const addedUser = ref('');

function startAddingUser() {
  addingUser.value = true;
  addedUser.value = '';
}

function addUser(email: any) {
  if (typeof email !== 'string') return;
  connected_users.push(email);
  addingUser.value = false;
  addedUser.value = email;
}
</script>

<style scoped>
.content {
  margin-bottom: 8rem;
}

.connected-user {
  display: flex;
  margin-bottom: 1rem;
}

.connected-user > p {
  border: 1px solid var(--color-main);
  border-radius: 8px;
  padding: 0.3rem 1rem;
  width: 30rem;
  max-width: 100%;
  margin-right: 2rem;
}

.connected-user > button {
  background-color: var(--red);
}

.success {
  display: flex;
  color: var(--green);
}

.success svg {
  margin-right: .5rem;
  margin-bottom: 1rem;
  width: 1rem;
}
</style>